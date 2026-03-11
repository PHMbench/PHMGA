from __future__ import annotations

from pathlib import Path
from typing import Dict


def build_case_artifact_paths(case_name: str, save_root: str | Path) -> Dict[str, str]:
    root = Path(save_root).resolve()
    case_dir = root / case_name
    return {
        "case_dir": str(case_dir),
        "run_dir": str(case_dir / "run"),
        "report_path": str(case_dir / "final_report.md"),
    }
