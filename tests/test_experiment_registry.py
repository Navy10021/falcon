"""실험 레지스트리 단위 테스트."""
import json
import os

import pytest

from utils.experiment_registry import ExperimentRegistry, RunRecord


def _write_manifest(base_dir, run_id, phase=1, seed=42, algorithm="ppo"):
    """헬퍼: 테스트용 run_manifest.json 생성"""
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "phase": phase,
        "algorithm": algorithm,
        "seed": seed,
        "episodes": 100,
        "checkpoint_dir": run_dir,
        "created_at": f"2026-01-01T00:00:{hash(run_id) % 60:02d}",
    }
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f)
    return path


def test_scan_finds_manifests(tmp_path):
    """scan should find all run_manifest.json files recursively."""
    _write_manifest(str(tmp_path), "run_01")
    _write_manifest(str(tmp_path), "run_02")
    _write_manifest(str(tmp_path), "run_03")

    registry = ExperimentRegistry(root_dir=str(tmp_path))
    found = registry.scan()
    assert found == 3
    assert registry.count == 3


def test_search_by_phase(tmp_path):
    """search(phase=) should filter correctly."""
    _write_manifest(str(tmp_path), "run_p1", phase=1)
    _write_manifest(str(tmp_path), "run_p2", phase=2)
    _write_manifest(str(tmp_path), "run_p3", phase=1)

    registry = ExperimentRegistry(root_dir=str(tmp_path))
    registry.scan()

    phase1_runs = registry.search(phase=1)
    assert len(phase1_runs) == 2
    assert all(r.phase == 1 for r in phase1_runs)


def test_search_by_algorithm(tmp_path):
    """search(algorithm=) should filter correctly."""
    _write_manifest(str(tmp_path), "run_a1", algorithm="ppo")
    _write_manifest(str(tmp_path), "run_a2", algorithm="mappo")

    registry = ExperimentRegistry(root_dir=str(tmp_path))
    registry.scan()

    mappo_runs = registry.search(algorithm="mappo")
    assert len(mappo_runs) == 1
    assert mappo_runs[0].algorithm == "mappo"


def test_latest(tmp_path):
    """latest() should return most recent run."""
    _write_manifest(str(tmp_path), "run_01")
    _write_manifest(str(tmp_path), "run_99")

    registry = ExperimentRegistry(root_dir=str(tmp_path))
    registry.scan()

    latest = registry.latest()
    assert latest is not None
    assert latest.run_id in ("run_01", "run_99")


def test_get_by_run_id(tmp_path):
    """get(run_id) should return exact match."""
    _write_manifest(str(tmp_path), "run_abc")

    registry = ExperimentRegistry(root_dir=str(tmp_path))
    registry.scan()

    record = registry.get("run_abc")
    assert record is not None
    assert record.run_id == "run_abc"

    assert registry.get("nonexistent") is None


def test_save_and_load_index(tmp_path):
    """save_index/load_index round trip should preserve records."""
    _write_manifest(str(tmp_path), "run_x1", phase=1)
    _write_manifest(str(tmp_path), "run_x2", phase=2)

    registry = ExperimentRegistry(root_dir=str(tmp_path))
    registry.scan()

    index_path = str(tmp_path / "index.json")
    registry.save_index(index_path)

    loaded = ExperimentRegistry.load_index(index_path)
    assert loaded.count == 2
    assert loaded.get("run_x1") is not None
    assert loaded.get("run_x2").phase == 2


def test_add_manual_record():
    """Manual add should register in index."""
    registry = ExperimentRegistry()
    record = RunRecord(run_id="manual_01", phase=1, seed=42)
    registry.add(record)

    assert registry.count == 1
    assert registry.get("manual_01") is not None


def test_summary(tmp_path):
    """summary should return valid structure."""
    _write_manifest(str(tmp_path), "run_s1", phase=1)
    _write_manifest(str(tmp_path), "run_s2", phase=2)

    registry = ExperimentRegistry(root_dir=str(tmp_path))
    registry.scan()

    s = registry.summary()
    assert s["total_runs"] == 2
    assert 1 in s["runs_by_phase"]
    assert 2 in s["runs_by_phase"]
