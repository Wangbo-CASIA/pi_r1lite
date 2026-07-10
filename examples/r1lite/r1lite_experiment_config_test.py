import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r1lite_experiment_config import apply_rollout_config


def _args(config_path, *, rtc=None):
    return argparse.Namespace(
        experiment=None,
        config=str(config_path),
        action_space="joint_delta",
        prompt=None,
        robot_server=None,
        policy_host=None,
        policy_port=None,
        control_hz=None,
        actions_per_infer=None,
        timeout=None,
        gripper_open_value=None,
        gripper_close_value=None,
        gripper_threshold=None,
        left_gripper_threshold=None,
        right_gripper_threshold=None,
        rtc=rtc,
        rtc_latency_k=None,
        rtc_inference_rate=None,
        rtc_max_guidance_weight=None,
    )


def test_apply_rollout_config_reads_rtc(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
experiment:
  prompt: test
robot:
  server_url: http://robot
rollout:
  policy_host: localhost
  policy_port: 8000
  control_hz: 10.0
  actions_per_infer: 5
  rtc:
    enabled: true
    latency_k: 4
    inference_rate: 3.0
    max_guidance_weight: 10.0
hg_dagger: {}
local: {}
action_spaces:
  joint_delta: {}
""",
        encoding="utf-8",
    )

    args = apply_rollout_config(_args(config))

    assert args.rtc is True
    assert args.rtc_latency_k == 4
    assert args.rtc_inference_rate == 3.0
    assert args.rtc_max_guidance_weight == 10.0


def test_apply_rollout_config_keeps_cli_rtc_override(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
experiment:
  prompt: test
robot:
  server_url: http://robot
rollout:
  rtc:
    enabled: true
    latency_k: 4
    inference_rate: 3.0
    max_guidance_weight: 10.0
hg_dagger: {}
local: {}
action_spaces:
  joint_delta: {}
""",
        encoding="utf-8",
    )

    args = apply_rollout_config(_args(config, rtc=False))

    assert args.rtc is False
