"""
experiment_registry.py
======================
실험 레지스트리 — run manifest 인덱스 + 검색.

여러 학습 실행(run)의 메타데이터를 통합 인덱스로 관리하고
Phase/알고리즘/시드/성능 기준으로 검색할 수 있게 한다.

사용 예시::

    registry = ExperimentRegistry(root_dir="checkpoints")
    registry.scan()  # 기존 run_manifest.json 파일 자동 수집

    # 검색
    results = registry.search(phase=1, min_win_rate=0.6)
    latest  = registry.latest(phase=2)

    # 인덱스 저장/로드
    registry.save_index("experiment_index.json")
    registry = ExperimentRegistry.load_index("experiment_index.json")

학술 연구용 합성 데이터
"""
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("falcon")


@dataclass
class RunRecord:
    """단일 실험 실행 레코드"""
    run_id: str
    phase: int
    algorithm: str = "ppo"
    seed: int = 0
    episodes: int = 0
    checkpoint_dir: str = ""
    config_path: Optional[str] = None
    git_sha: Optional[str] = None
    created_at: str = ""
    # 성능 메트릭 (사후 기록 가능)
    win_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    # manifest 파일 경로
    manifest_path: str = ""

    @classmethod
    def from_manifest(cls, manifest_path: str) -> "RunRecord":
        """run_manifest.json에서 레코드 생성"""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            run_id=data.get("run_id", ""),
            phase=data.get("phase", 0),
            algorithm=data.get("algorithm", "ppo"),
            seed=data.get("seed", 0),
            episodes=data.get("episodes", 0),
            checkpoint_dir=data.get("checkpoint_dir", ""),
            config_path=data.get("config_path"),
            git_sha=data.get("git_sha"),
            created_at=data.get("created_at", ""),
            win_rate=data.get("win_rate"),
            avg_reward=data.get("avg_reward"),
            manifest_path=manifest_path,
        )


class ExperimentRegistry:
    """
    실험 레지스트리.

    파일 시스템 내 run_manifest.json 파일을 스캔하여
    통합 인덱스를 구축하고 검색 기능을 제공한다.
    """

    def __init__(self, root_dir: str = "checkpoints"):
        self.root_dir = root_dir
        self.records: List[RunRecord] = []
        self._index: Dict[str, RunRecord] = {}  # run_id → record

    def scan(self, pattern: str = "**/run_manifest.json") -> int:
        """
        root_dir 내 모든 run_manifest.json을 스캔하여 인덱스 구축.

        Returns: 새로 추가된 레코드 수
        """
        added = 0
        search_pattern = os.path.join(self.root_dir, pattern)
        for path in glob.glob(search_pattern, recursive=True):
            try:
                record = RunRecord.from_manifest(path)
                if record.run_id and record.run_id not in self._index:
                    self.records.append(record)
                    self._index[record.run_id] = record
                    added += 1
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("Failed to parse manifest %s: %s", path, e)
        # 시간순 정렬
        self.records.sort(key=lambda r: r.created_at, reverse=True)
        return added

    def add(self, record: RunRecord):
        """레코드 수동 추가"""
        if record.run_id in self._index:
            # 기존 레코드 업데이트
            self._index[record.run_id] = record
            self.records = [r if r.run_id != record.run_id else record for r in self.records]
        else:
            self.records.insert(0, record)
            self._index[record.run_id] = record

    def search(
        self,
        phase: Optional[int] = None,
        algorithm: Optional[str] = None,
        seed: Optional[int] = None,
        min_win_rate: Optional[float] = None,
        max_results: int = 50,
    ) -> List[RunRecord]:
        """조건 기반 검색"""
        results = []
        for r in self.records:
            if phase is not None and r.phase != phase:
                continue
            if algorithm is not None and r.algorithm != algorithm:
                continue
            if seed is not None and r.seed != seed:
                continue
            if min_win_rate is not None and (r.win_rate is None or r.win_rate < min_win_rate):
                continue
            results.append(r)
            if len(results) >= max_results:
                break
        return results

    def latest(self, phase: Optional[int] = None) -> Optional[RunRecord]:
        """최신 실행 레코드 조회"""
        results = self.search(phase=phase, max_results=1)
        return results[0] if results else None

    def get(self, run_id: str) -> Optional[RunRecord]:
        """run_id로 직접 조회"""
        return self._index.get(run_id)

    @property
    def count(self) -> int:
        return len(self.records)

    def summary(self) -> Dict[str, Any]:
        """레지스트리 요약"""
        phases = {}
        for r in self.records:
            phases.setdefault(r.phase, 0)
            phases[r.phase] += 1
        return {
            "total_runs": self.count,
            "runs_by_phase": phases,
            "root_dir": self.root_dir,
        }

    def save_index(self, path: str):
        """인덱스를 JSON으로 저장"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "version": "falcon-experiment-registry-v1",
            "root_dir": self.root_dir,
            "created_at": datetime.now().isoformat(),
            "records": [asdict(r) for r in self.records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_index(cls, path: str) -> "ExperimentRegistry":
        """저장된 인덱스에서 레지스트리 복원"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        registry = cls(root_dir=data.get("root_dir", "checkpoints"))
        for rec_data in data.get("records", []):
            record = RunRecord(**{k: v for k, v in rec_data.items() if k in RunRecord.__dataclass_fields__})
            registry.records.append(record)
            registry._index[record.run_id] = record
        return registry
