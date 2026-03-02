"""
security.py
===========
보안 유틸리티 — 입력 검증 및 경로 안전화

학술 연구용 합성 데이터
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


# ──────────────────────────────────────────────
# Checkpoint Path Validation
# ──────────────────────────────────────────────

# 허용된 체크포인트 확장자
_ALLOWED_CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".bin", ".ckpt"}

# 체크포인트 검색 허용 디렉토리 (기본값)
_DEFAULT_ALLOWED_DIRS = ["checkpoints", "runs", "outputs", "/tmp"]


def validate_checkpoint_path(
    path: str,
    allowed_dirs: Optional[List[str]] = None,
    strict: bool = False,
) -> str:
    """
    체크포인트 경로 검증.

    Parameters
    ----------
    path : str
        검증할 체크포인트 경로
    allowed_dirs : list[str], optional
        허용된 상위 디렉토리. None이면 기본 목록 사용.
    strict : bool
        True이면 allowed_dirs 외부 경로를 거부.
        False이면 경고만 출력.

    Returns
    -------
    str : 정규화된 절대 경로

    Raises
    ------
    ValueError : 확장자가 허용되지 않는 경우
    PermissionError : strict 모드에서 허용 디렉토리 외부인 경우
    FileNotFoundError : 파일이 존재하지 않는 경우
    """
    # 경로 정규화 (symlink 해석, .. 제거)
    resolved = str(Path(path).resolve())

    # 1. 확장자 검증
    ext = Path(resolved).suffix.lower()
    if ext not in _ALLOWED_CHECKPOINT_EXTENSIONS:
        raise ValueError(
            f"Invalid checkpoint extension '{ext}'. "
            f"Allowed: {_ALLOWED_CHECKPOINT_EXTENSIONS}"
        )

    # 2. 파일 존재 확인
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    # 3. 디렉토리 allowlist 검증 (strict 모드)
    if strict:
        dirs = allowed_dirs or _DEFAULT_ALLOWED_DIRS
        abs_dirs = [str(Path(d).resolve()) for d in dirs]
        if not any(resolved.startswith(d) for d in abs_dirs):
            raise PermissionError(
                f"Checkpoint path '{resolved}' is outside allowed directories: {dirs}"
            )

    return resolved


def safe_yaml_load(path: str) -> dict:
    """
    안전한 YAML 로딩 (yaml.safe_load 강제).

    yaml.load(Loader=FullLoader) 등 unsafe loader를 방지.
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
