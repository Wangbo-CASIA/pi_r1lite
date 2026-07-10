#!/usr/bin/env python3
"""Train an ALOE Q-chunking critic on replay collected from R1Lite."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from openpi_client import websocket_client_policy
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from openpi.aloe import schema
from openpi.aloe.critic import AloeCriticConfig
from openpi.aloe.critic import AloeQCritic
from openpi.aloe.critic import discounted_chunk_return
from openpi.aloe.critic import polyak_update
from openpi.aloe.dataset import AloeReplayDataset


def _to_torch(batch: dict, device: torch.device) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    return {
        "state": torch.as_tensor(batch["compact_state"], dtype=torch.float32, device=device),
        "next_state": torch.as_tensor(batch["next_compact_state"], dtype=torch.float32, device=device),
        "images": {
            key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in batch["images"].items()
        },
        "next_images": {
            key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in batch["next_images"].items()
        },
        "actions": torch.as_tensor(batch["actions"], dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device),
        "dones": torch.as_tensor(batch["dones"], dtype=torch.float32, device=device),
        "valid_mask": torch.as_tensor(batch["valid_mask"], dtype=torch.float32, device=device),
    }


def _sample_next_actions(
    policy: websocket_client_policy.WebsocketClientPolicy,
    batch: dict,
    *,
    action_horizon: int,
) -> np.ndarray:
    actions = []
    for index, sample in enumerate(batch["samples"]):
        obs_arrays = batch["_dataset"].load_observation_arrays(sample.next_observation, sample.run_id)
        obs = {
            "images": obs_arrays["images"],
            "state": obs_arrays["state"],
            "prompt": sample.prompt,
        }
        result = policy.infer(obs)
        if "actions" not in result:
            raise RuntimeError(f"policy response missing actions. Keys: {sorted(result)}")
        action = np.asarray(result["actions"], dtype=np.float32)
        if action.shape != (action_horizon, schema.ACTION_DIM):
            raise RuntimeError(
                f"policy next action shape mismatch for sample {index}: expected {(action_horizon, schema.ACTION_DIM)}, got {action.shape}"
            )
        actions.append(action)
    return np.stack(actions)


def _target_q(
    critic_target: AloeQCritic,
    batch_np: dict,
    batch: dict,
    next_actions: np.ndarray,
    *,
    gamma: float,
) -> torch.Tensor:
    device = next(critic_target.parameters()).device
    bootstrap_batch = {
        "state": batch["next_state"],
        "actions": torch.as_tensor(next_actions, dtype=torch.float32, device=device),
        "images": batch["next_images"],
    }
    with torch.no_grad():
        q_next = critic_target(bootstrap_batch).min(dim=1).values
        returns = discounted_chunk_return(batch["rewards"], batch["valid_mask"], gamma)
        full_chunk = batch["valid_mask"][:, -1]
        bootstrap_mask = (1.0 - batch["dones"]) * full_chunk
        discount_h = float(gamma) ** int(batch["valid_mask"].shape[1])
        return returns + discount_h * bootstrap_mask * q_next


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--pi05-base-checkpoint", default=None)
    parser.add_argument("--vision-mode", choices=("openpi_pi05_siglip", "tiny_siglip_for_test"), default="openpi_pi05_siglip")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    device = torch.device(args.device)
    dataset = AloeReplayDataset(args.replay_root, action_horizon=args.action_horizon)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    config = AloeCriticConfig(
        action_horizon=args.action_horizon,
        ensemble=args.ensemble,
        embed_dim=args.embed_dim,
        layers=args.layers,
        heads=args.heads,
        pi05_base_checkpoint=args.pi05_base_checkpoint,
        vision_mode=args.vision_mode,
    )
    critic = AloeQCritic(config).to(device)
    target = AloeQCritic(config).to(device)
    target.load_state_dict(critic.state_dict())
    optimizer = AdamW(critic.parameters(), lr=args.lr, betas=(0.9, 0.95))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for step in trange(args.steps, dynamic_ncols=True):
        batch_np = dataset.torch_batch(args.batch_size)
        batch_np["_dataset"] = dataset
        batch = _to_torch(batch_np, device)
        next_actions = _sample_next_actions(policy, batch_np, action_horizon=args.action_horizon)
        y = _target_q(target, batch_np, batch, next_actions, gamma=args.gamma)
        q = critic({"state": batch["state"], "actions": batch["actions"], "images": batch["images"]})
        loss = torch.mean((q - y.unsqueeze(1)) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"critic loss became non-finite at step {step}: {float(loss.detach().cpu())}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        polyak_update(target, critic, args.polyak)
        if (step + 1) % args.save_interval == 0 or step == args.steps - 1:
            torch.save(
                {
                    "config": dataclasses.asdict(config),
                    "critic": critic.state_dict(),
                    "target": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                },
                args.output_dir / f"critic_step_{step + 1}.pt",
            )
        if step % 50 == 0:
            print(
                f"step={step} loss={float(loss.detach().cpu()):.6f} "
                f"q_mean={float(q.detach().mean().cpu()):.4f} target_mean={float(y.detach().mean().cpu()):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
