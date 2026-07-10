import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import convert_r1lite_raw_to_lerobot_joint as converter


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_prepare_source_manifest_backfills_existing_episodes(tmp_path):
    manifest = tmp_path / "source_episodes.jsonl"
    raw_path = tmp_path / "recovery_0_RAW"

    converter._prepare_source_manifest(  # noqa: SLF001
        manifest,
        existing_episode_count=2,
        input_dirs=[raw_path],
        source_label="hgdagger",
        allow_duplicate_source=False,
        skip_existing_source=False,
    )
    converter._append_source_record(manifest, 2, "hgdagger", raw_path)  # noqa: SLF001

    assert _read_jsonl(manifest) == [
        {"episode_index": 0, "source": "base_demo", "raw_path": None},
        {"episode_index": 1, "source": "base_demo", "raw_path": None},
        {"episode_index": 2, "source": "hgdagger", "raw_path": str(raw_path.resolve())},
    ]


def test_prepare_source_manifest_rejects_duplicate_raw_paths(tmp_path):
    manifest = tmp_path / "source_episodes.jsonl"
    raw_path = tmp_path / "recovery_0_RAW"
    manifest.write_text(
        json.dumps({"episode_index": 0, "source": "hgdagger", "raw_path": str(raw_path.resolve())}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate RAW source paths"):
        converter._prepare_source_manifest(  # noqa: SLF001
            manifest,
            existing_episode_count=1,
            input_dirs=[raw_path],
            source_label="hgdagger",
            allow_duplicate_source=False,
            skip_existing_source=False,
        )


def test_prepare_source_manifest_allows_existing_raw_paths_when_skipping(tmp_path):
    manifest = tmp_path / "source_episodes.jsonl"
    raw_path = tmp_path / "recovery_0_RAW"
    manifest.write_text(
        json.dumps({"episode_index": 0, "source": "hgdagger", "raw_path": str(raw_path.resolve())}) + "\n",
        encoding="utf-8",
    )

    existing = converter._prepare_source_manifest(  # noqa: SLF001
        manifest,
        existing_episode_count=1,
        input_dirs=[raw_path],
        source_label="hgdagger",
        allow_duplicate_source=False,
        skip_existing_source=True,
    )

    assert existing == {str(raw_path.resolve())}


def test_prepare_source_manifest_rejects_row_count_mismatch(tmp_path):
    manifest = tmp_path / "source_episodes.jsonl"
    manifest.write_text(
        json.dumps({"episode_index": 0, "source": "base_demo", "raw_path": None}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset has 2 episodes"):
        converter._prepare_source_manifest(  # noqa: SLF001
            manifest,
            existing_episode_count=2,
            input_dirs=[tmp_path / "recovery_0_RAW"],
            source_label="hgdagger",
            allow_duplicate_source=False,
            skip_existing_source=False,
        )


def test_build_joint_episode_samples_does_not_require_eef_topics(monkeypatch, tmp_path):
    topics_seen = []

    def fake_read_topic_series(_input_dir, topics):
        topics_seen.extend(topics)
        return {topic: topic for topic in topics}

    monkeypatch.setattr(converter, "read_topic_series", fake_read_topic_series)
    monkeypatch.setattr(converter, "overlapping_timeline", lambda _series_map, _fps: [100])
    monkeypatch.setattr(converter, "nearest_value_at", lambda series, _timestamp: f"nearest:{series}")

    def fake_value_at(series, _timestamp):
        if "feedback_arm" in series:
            return {"position": [0.0] * 6}
        return [100.0]

    monkeypatch.setattr(converter, "value_at", fake_value_at)

    samples = converter._build_joint_episode_samples(tmp_path / "episode_RAW", 10, dict(converter.DEFAULT_TOPICS))  # noqa: SLF001

    assert converter.DEFAULT_TOPICS["left_tcp_pose"] not in topics_seen
    assert converter.DEFAULT_TOPICS["right_tcp_pose"] not in topics_seen
    assert samples == [
        {
            "timestamp_ns": 100,
            "head": f"nearest:{converter.DEFAULT_TOPICS['head']}",
            "left_wrist": f"nearest:{converter.DEFAULT_TOPICS['left_wrist']}",
            "right_wrist": f"nearest:{converter.DEFAULT_TOPICS['right_wrist']}",
            "left_joint": {"position": [0.0] * 6},
            "right_joint": {"position": [0.0] * 6},
            "left_gripper": [100.0],
            "right_gripper": [100.0],
        }
    ]
