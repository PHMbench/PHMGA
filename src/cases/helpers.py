from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def build_case_paths(case_name: str, save_root: str | Path) -> Dict[str, str]:
    root = Path(save_root).resolve()
    case_dir = root / case_name
    return {
        "case_dir": str(case_dir),
        "resolved_config_path": str(case_dir / "resolved_config.json"),
        "metadata_snapshot_path": str(case_dir / "metadata_snapshot.json"),
    }


def write_metadata_snapshot(snapshot: Dict[str, Any], output_path: str | Path) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
