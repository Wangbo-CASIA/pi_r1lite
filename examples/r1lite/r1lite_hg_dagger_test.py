import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r1lite_hg_dagger as hg_dagger


def _record_args() -> argparse.Namespace:
    return argparse.Namespace(
        record_output_dir="/home/r1lite/GalaxeaDataset/hgdagger/r1lite_pack_phone_new_state",
        robot_server="http://robot:8001",
        timeout=2.0,
        record_stop_timeout=60.0,
        policy_host="localhost",
        policy_port=8000,
        prompt="test task",
        max_steps=120,
        control_hz=10.0,
        actions_per_infer=10,
        record_topics=["/topic"],
    )


def test_recording_session_reuses_one_rollout_output_dir(monkeypatch):
    monkeypatch.setattr(hg_dagger, "_record_session_dir_name", lambda: "session_20260608_153012_123")
    calls = []

    def fake_request_json(method, url, timeout, payload=None):
        calls.append((method, url, timeout, payload))
        if url.endswith("/record/start"):
            return {"episode_stem": f"recovery_{len([call for call in calls if call[1].endswith('/record/start')])}"}
        return {}

    monkeypatch.setattr(hg_dagger, "_request_json", fake_request_json)
    args = _record_args()
    session = hg_dagger.InterventionRecordingSession(args, "scripts/run_r1lite_openpi_policy.py")

    session.start(source="tabletop", rollout_step=10, seq=10, arm="dual")
    session.save(rollout_step=20, seq=21, end_reason="operator_save")
    session.start(source="tabletop", rollout_step=40, seq=42, arm="dual")
    session.discard(rollout_step=50, seq=53, end_reason="operator_discard")

    expected_output_dir = (
        "/home/r1lite/GalaxeaDataset/hgdagger/r1lite_pack_phone_new_state/session_20260608_153012_123"
    )
    start_payloads = [payload for _method, url, _timeout, payload in calls if url.endswith("/record/start")]

    assert session.record_output_dir == expected_output_dir
    assert args.record_output_dir == "/home/r1lite/GalaxeaDataset/hgdagger/r1lite_pack_phone_new_state"
    assert [payload["output_dir"] for payload in start_payloads] == [expected_output_dir, expected_output_dir]


def test_record_session_output_dir_handles_robot_root_dir():
    assert hg_dagger._record_session_output_dir("/", "session_20260608_153012_123") == (  # noqa: SLF001
        "/session_20260608_153012_123"
    )
