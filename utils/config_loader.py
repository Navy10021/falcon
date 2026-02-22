"""
utils/config_loader.py
======================
YAML 설정 파일 로더 — CLI 인자로 override 지원

사용 예:
    cfg = load_config("configs/default.yaml", args, section="training")
    # args의 non-None 값이 YAML 값을 덮어씀
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _load_yaml(path: str) -> Dict[str, Any]:
    """YAML 파일 로드 (PyYAML 없으면 간이 파서 사용)."""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # PyYAML 미설치 시 간이 key: value 파서 (중첩 지원)
        return _simple_yaml_load(path)


def _simple_yaml_load(path: str) -> Dict[str, Any]:
    """PyYAML 없이 동작하는 단순 YAML 파서 (depth-1 섹션 + scalar 값)."""
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.lstrip()

            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(stripped)

            if indent == 0 and stripped.endswith(":"):
                # 최상위 섹션
                current_section = stripped[:-1]
                result[current_section] = {}
            elif indent > 0 and ":" in stripped:
                key, _, raw_val = stripped.partition(":")
                key = key.strip()
                raw_val = raw_val.strip()

                # 인라인 주석 제거
                if "#" in raw_val:
                    raw_val = raw_val[: raw_val.index("#")].strip()

                val: Any = _parse_scalar(raw_val)

                if current_section:
                    result[current_section][key] = val
                else:
                    result[key] = val

    return result


def _parse_scalar(s: str) -> Any:
    """문자열을 Python 기본형으로 변환."""
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~", ""):
        return None
    # 따옴표 문자열
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    # 정수
    try:
        return int(s)
    except ValueError:
        pass
    # 부동소수점 (지수 표기 포함)
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_config(
    config_path: Optional[str],
    args=None,
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """
    YAML 설정 로드 후 argparse Namespace의 non-None 값으로 override.

    Parameters
    ----------
    config_path : str | None
        YAML 파일 경로. None이면 빈 dict 반환.
    args : argparse.Namespace | None
        CLI 파싱 결과. None이면 YAML 값만 사용.
    section : str | None
        YAML 최상위 섹션 키 (예: "training"). None이면 전체 dict 사용.

    Returns
    -------
    dict
        최종 설정 dict. args 속성명 기준으로 하이픈(-) → 언더스코어(_) 변환.
    """
    raw: Dict[str, Any] = {}

    if config_path and os.path.isfile(config_path):
        full = _load_yaml(config_path)
        raw = full.get(section, full) if section else full

    # CLI override: args에 명시된 값이 있으면 YAML 값을 덮어씀
    if args is not None:
        for key, val in vars(args).items():
            if val is not None and key != "config":
                # argparse는 하이픈을 언더스코어로 저장하므로 그대로 사용
                raw[key] = val

    return raw


def apply_config_to_args(args, config_path: Optional[str], section: Optional[str] = None):
    """
    YAML config를 args Namespace에 기본값으로 주입.
    CLI에서 명시적으로 지정된 값은 유지.

    사용 예:
        apply_config_to_args(args, args.config, section="training")
    """
    if not config_path or not os.path.isfile(config_path):
        return

    full = _load_yaml(config_path)
    cfg = full.get(section, full) if section else full

    for key, yaml_val in cfg.items():
        # argparse default가 None이거나 해당 키가 없는 경우에만 YAML 값 적용
        current = getattr(args, key, None)
        if current is None:
            setattr(args, key, yaml_val)
