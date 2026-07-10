#!/usr/bin/env python3
"""Train an OpenPI actor with ALOE advantage-weighted flow matching."""

from __future__ import annotations

import argparse
import dataclasses
import functools
from pathlib import Path
import sys
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import websocket_client_policy
import optax
import torch
from tqdm import trange

from openpi import transforms as _transforms
from openpi.aloe import schema
from openpi.aloe.critic import AloeCriticConfig
from openpi.aloe.critic import AloeQCritic
from openpi.aloe.dataset import AloeReplayDataset
from openpi.models import model as _model
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import sharding
from openpi.training import weight_loaders
from openpi.training import optimizer as _optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train as train_lib  # noqa: E402


class _AloeDataLoaderForCheckpoint:
    def __init__(self, data_config: _config.DataConfig):
        self._data_config = data_config

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        raise NotImplementedError("ALOE actor data loader is not iterable; it only provides checkpoint assets")


def _params_path(checkpoint: Path) -> Path:
    if (checkpoint / "params").exists():
        return checkpoint / "params"
    if checkpoint.name == "params" and checkpoint.exists():
        return checkpoint
    raise FileNotFoundError(f"actor checkpoint must be a step directory containing params, or a params dir: {checkpoint}")


def _assets_dir(checkpoint: Path) -> Path:
    if (checkpoint / "assets").exists():
        return checkpoint / "assets"
    if checkpoint.name == "params" and (checkpoint.parent / "assets").exists():
        return checkpoint.parent / "assets"
    raise FileNotFoundError(f"actor checkpoint assets directory not found near: {checkpoint}")


def _load_critic(path: Path, device: torch.device) -> AloeQCritic:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = AloeCriticConfig(**payload["config"])
    critic = AloeQCritic(config).to(device)
    critic.load_state_dict(payload["critic"])
    critic.eval()
    return critic


def _torch_critic_batch(batch_np: dict[str, Any], *, actions: np.ndarray, device: torch.device) -> dict[str, Any]:
    return {
        "state": torch.as_tensor(batch_np["compact_state"], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.float32, device=device),
        "images": {
            key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in batch_np["images"].items()
        },
    }


def _policy_actions_for_samples(
    policy: websocket_client_policy.WebsocketClientPolicy,
    dataset: AloeReplayDataset,
    samples,
    *,
    action_horizon: int,
) -> np.ndarray:
    actions = []
    for index, sample in enumerate(samples):
        obs_arrays = dataset.load_observation_arrays(sample.observation, sample.run_id)
        result = policy.infer(
            {
                "images": obs_arrays["images"],
                "state": obs_arrays["state"],
                "prompt": sample.prompt,
            }
        )
        action = np.asarray(result.get("actions"), dtype=np.float32)
        if action.shape != (action_horizon, schema.ACTION_DIM):
            raise RuntimeError(
                f"policy action shape mismatch for sample {index}: expected {(action_horizon, schema.ACTION_DIM)}, got {action.shape}"
            )
        actions.append(action)
    return np.stack(actions)


def _critic_weights(
    critic: AloeQCritic,
    policy: websocket_client_policy.WebsocketClientPolicy,
    dataset: AloeReplayDataset,
    batch_np: dict[str, Any],
    *,
    beta: float,
    adv_clip: float,
    device: torch.device,
) -> np.ndarray:
    samples = batch_np["samples"]
    data_actions = np.asarray(batch_np["actions"], dtype=np.float32)
    policy_actions = _policy_actions_for_samples(policy, dataset, samples, action_horizon=data_actions.shape[1])
    with torch.no_grad():
        q_data = critic(_torch_critic_batch(batch_np, actions=data_actions, device=device)).min(dim=1).values
        q_policy = critic(_torch_critic_batch(batch_np, actions=policy_actions, device=device)).min(dim=1).values
        advantage = q_data - q_policy
        weights = torch.exp(torch.clamp(advantage / float(beta), min=-float(adv_clip), max=float(adv_clip)))
    weights_np = weights.detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(weights_np)):
        raise RuntimeError("ALOE actor weights contain non-finite values")
    return weights_np


def _build_transform(data_config: _config.DataConfig):
    return _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _actor_batch(
    dataset: AloeReplayDataset,
    data_config: _config.DataConfig,
    batch_np: dict[str, Any],
) -> tuple[_model.Observation, _model.Actions, np.ndarray]:
    transform = _build_transform(data_config)
    transformed = []
    for sample in batch_np["samples"]:
        obs_arrays = dataset.load_observation_arrays(sample.observation, sample.run_id)
        data = {
            "images": obs_arrays["images"],
            "state": obs_arrays["state"],
            "actions": sample.actions,
            "prompt": sample.prompt,
        }
        transformed.append(transform(data))
    batch = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *transformed)
    actions = np.asarray(batch.pop("actions"), dtype=np.float32)
    valid_mask = np.asarray(batch_np["valid_mask"], dtype=np.float32)
    return _model.Observation.from_dict(batch), actions, valid_mask


def _train_step(config, rng, state, observation, actions, valid_mask, weights):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions, valid_mask, weights):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        weights = weights[:, None]
        denom = jnp.maximum(jnp.sum(valid_mask), 1.0)
        return jnp.sum(chunked_loss * valid_mask * weights) / denom

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model,
        train_rng,
        observation,
        actions,
        valid_mask,
        weights,
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state)
    return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-config", required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-host", required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.beta <= 0:
        parser.error("--beta must be positive")
    if args.batch_size <= 0 or args.steps <= 0:
        parser.error("--batch-size and --steps must be positive")

    actor_checkpoint = args.actor_checkpoint.resolve()
    params_path = _params_path(actor_checkpoint)
    assets_dir = _assets_dir(actor_checkpoint)
    train_config = _config.get_config(args.actor_config)
    train_config = dataclasses.replace(
        train_config,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params_path)),
        exp_name=args.output_dir.name,
        checkpoint_base_dir=str(args.output_dir.parent.parent if args.output_dir.parent.name == train_config.name else args.output_dir.parent),
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        lr_schedule=(
            _optimizer.CosineDecaySchedule(
                warmup_steps=min(100, max(args.steps - 1, 1)),
                peak_lr=float(args.lr),
                decay_steps=max(args.steps, 1),
                decay_lr=float(args.lr) * 0.1,
            )
            if args.lr is not None
            else train_config.lr_schedule
        ),
        overwrite=args.overwrite,
        wandb_enabled=False,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("actor data config must have an asset_id to load normalization stats")
    norm_stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    dataset = AloeReplayDataset(args.replay_root, action_horizon=args.action_horizon)
    critic = _load_critic(args.critic_checkpoint, torch.device(args.device))
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)

    mesh = sharding.make_mesh(train_config.fsdp_devices)
    rng = jax.random.key(train_config.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = train_lib.init_train_state(train_config, init_rng, mesh, resume=False)
    ptrain_step = jax.jit(functools.partial(_train_step, train_config), donate_argnums=(1,))

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        args.output_dir,
        keep_period=None,
        overwrite=args.overwrite,
        resume=False,
    )
    data_loader_for_ckpt = _AloeDataLoaderForCheckpoint(data_config)

    for step in trange(args.steps, dynamic_ncols=True):
        batch_np = dataset.torch_batch(args.batch_size)
        weights = _critic_weights(
            critic,
            policy,
            dataset,
            batch_np,
            beta=args.beta,
            adv_clip=args.adv_clip,
            device=torch.device(args.device),
        )
        observation, actions, valid_mask = _actor_batch(dataset, data_config, batch_np)
        observation = jax.tree.map(jnp.asarray, observation)
        actions = jnp.asarray(actions)
        valid_mask = jnp.asarray(valid_mask)
        weights_jax = jnp.asarray(weights)
        state, info = ptrain_step(train_rng, state, observation, actions, valid_mask, weights_jax)
        if step % 20 == 0:
            info_np = jax.device_get(info)
            print(
                f"step={step} loss={float(info_np['loss']):.6f} "
                f"grad_norm={float(info_np['grad_norm']):.4f} weight_mean={float(weights.mean()):.4f}",
                flush=True,
            )
        if (step + 1) % args.save_interval == 0 or step == args.steps - 1:
            _checkpoints.save_state(checkpoint_manager, state, data_loader_for_ckpt, step + 1)

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
