#!/usr/bin/env python3
"""
vLLM>=0.9 may raise:
  ValueError: Found conflicts between 'rope_type=default' ... and 'type=mrope' ...

when HF config.json still has both modern ``rope_type`` and legacy ``type`` under
``rope_scaling`` (often on Qwen2.5-VL trees). This script removes the legacy
``type`` key when it disagrees with ``rope_type`` (same fix for nested
``text_config`` if present).

Usage:
  python src/scripts/fix_hf_rope_scaling_for_vllm.py /path/to/model1 /path/to/model2 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _fix_rope_scaling_dict(rs: dict[str, Any]) -> bool:
    """
    Normalize rope_scaling to avoid vLLM conflicts.

    vLLM mutates rope_type in-place for legacy values like 'mrope' -> 'default'.
    If the legacy 'type' key is left behind (e.g. type='mrope'), vLLM can raise
    when it later sees rope_type='default' and type='mrope'.

    Strategy:
    - If only legacy 'type' exists: copy to rope_type and delete 'type'
    - If both exist: always delete legacy 'type' (regardless of equality)
    """
    # If legacy mrope is present, normalize to transformers/vLLM-compatible rope_type='default'
    # while keeping mrope_section.
    if rs.get("rope_type") == "mrope" or rs.get("type") == "mrope":
        rs["rope_type"] = "default"
        if "type" in rs:
            del rs["type"]
        return True

    if "type" not in rs:
        return False
    if "rope_type" not in rs:
        rs["rope_type"] = rs["type"]
        del rs["type"]
        return True
    # Both exist; drop legacy field to prevent future divergence.
    del rs["type"]
    return True


def _fix_config_tree(cfg: dict[str, Any]) -> int:
    n = 0
    rs = cfg.get("rope_scaling")
    if isinstance(rs, dict) and _fix_rope_scaling_dict(rs):
        n += 1
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        trs = tc.get("rope_scaling")
        if isinstance(trs, dict) and _fix_rope_scaling_dict(trs):
            n += 1
    return n


def fix_one_model_dir(model_dir: Path) -> int:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        print(f"[skip] no config.json: {model_dir}", file=sys.stderr)
        return 0
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        return 0
    n = _fix_config_tree(cfg)
    if n:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[ok] fixed {n} rope_scaling block(s) in {cfg_path}")
    else:
        print(f"[ok] no rope conflict in {cfg_path}")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Strip legacy rope_scaling.type when it conflicts with rope_type (vLLM).")
    p.add_argument("model_dirs", nargs="+", type=str, help="HF model directories containing config.json")
    args = p.parse_args()
    total = 0
    for d in args.model_dirs:
        total += fix_one_model_dir(Path(d).resolve())
    print(f"[done] dirs={len(args.model_dirs)} patches_written={total}")


if __name__ == "__main__":
    main()
