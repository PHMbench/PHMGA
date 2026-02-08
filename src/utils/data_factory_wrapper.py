from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def _default_vibench_code_root() -> str:
    return os.environ.get(
        "PHM_VIBENCH_CODE_ROOT",
        "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2",
    )


def _ensure_vibench_on_syspath(code_root: str) -> None:
    root = str(Path(code_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _ensure_pytorch_lightning_stub() -> None:
    """PHM-Vibench Default_dataset imports `pytorch_lightning.utilities.CombinedLoader`.

    For minimal environments where pytorch_lightning isn't installed, inject a tiny stub.
    The DataFactory path we use here does not depend on CombinedLoader at runtime.
    """
    try:  # pragma: no cover
        import pytorch_lightning  # noqa: F401
        return
    except Exception:
        pass

    import types

    pl = types.ModuleType("pytorch_lightning")
    utilities = types.ModuleType("pytorch_lightning.utilities")

    class CombinedLoader:  # noqa: D401 - minimal stub
        def __init__(self, loaders: Any, *args: Any, **kwargs: Any):
            self.loaders = loaders

    utilities.CombinedLoader = CombinedLoader  # type: ignore[attr-defined]
    pl.utilities = utilities  # type: ignore[attr-defined]
    sys.modules.setdefault("pytorch_lightning", pl)
    sys.modules.setdefault("pytorch_lightning.utilities", utilities)


def _read_metadata_table(path: Path):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "pandas is required to read PHM-Vibench metadata. Please install pandas."
        ) from e

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _write_filtered_metadata(
    *,
    data_dir: Path,
    metadata_path: Path,
    dataset_name: str,
) -> str:
    df = _read_metadata_table(metadata_path)
    if "Name" not in df.columns:
        raise ValueError("Metadata file must contain a 'Name' column for dataset selection.")

    filtered = df[df["Name"].astype(str) == str(dataset_name)].copy()
    if filtered.empty:
        candidates = sorted({str(x) for x in df["Name"].dropna().unique().tolist()})
        raise ValueError(f"dataset_name={dataset_name!r} not found in metadata Name. candidates[:20]={candidates[:20]}")

    out_dir = data_dir / ".phmga_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metadata_filtered_{dataset_name}.csv"
    filtered.to_csv(out_path, index=False)
    return str(out_path.relative_to(data_dir))


@dataclass(frozen=True)
class VibenchBuildResult:
    train_loader: Any
    val_loader: Any
    test_loader: Any
    label_to_index: Dict[str, int]


class _WrappedLoader:
    def __init__(
        self,
        base_loader: Any,
        *,
        label_to_index: Dict[str, int],
        window_size: Optional[int] = None,
    ):
        self._base = base_loader
        self._label_to_index = dict(label_to_index)
        self._window_size = int(window_size) if window_size is not None else None

    def __len__(self) -> int:  # pragma: no cover
        try:
            return len(self._base)
        except Exception:
            return 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import torch

        for batch in self._base:
            if not isinstance(batch, dict):
                raise ValueError(f"Expected vibench batch dict, got: {type(batch).__name__}")
            if "x" not in batch or "y" not in batch:
                raise ValueError(f"Expected keys x/y in vibench batch, got keys={list(batch.keys())}")

            x = batch["x"]
            y = batch["y"]
            file_id = batch.get("file_id")

            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            x = x.to(dtype=torch.float32)

            if x.ndim == 2:
                x = x.unsqueeze(-1)
            if x.ndim != 3:
                raise ValueError(f"Expected x shape (B,L,C), got {tuple(x.shape)}")

            if self._window_size is not None and x.shape[1] != self._window_size and x.shape[2] == self._window_size:
                # Handle (B,C,L) -> (B,L,C) when window_size is known.
                x = x.permute(0, 2, 1).contiguous()

            # y: map to class indices
            if isinstance(y, torch.Tensor):
                y_list = y.detach().cpu().tolist()
            else:
                y_list = list(y) if isinstance(y, (list, tuple)) else [y]
            y_idx = [self._label_to_index[str(v)] for v in y_list]
            y_t = torch.tensor(y_idx, dtype=torch.int64)

            # file_id: ensure list[str]
            if file_id is None:
                file_ids: List[str] = [f"sample_{i}" for i in range(int(x.shape[0]))]
            elif isinstance(file_id, (list, tuple)):
                file_ids = [str(v) for v in file_id]
            elif hasattr(file_id, "tolist"):
                file_ids = [str(v) for v in file_id.tolist()]
            else:
                file_ids = [str(file_id) for _ in range(int(x.shape[0]))]

            yield {"x": x, "y": y_t, "file_id": file_ids}


class PHMVibenchDataFactory:
    """Thin wrapper over PHM-Vibench `src.data_factory.build_data`."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self._built: VibenchBuildResult | None = None

    def build(self) -> VibenchBuildResult:
        if self._built is not None:
            return self._built

        backend = str(self.cfg.get("backend") or "vibench").lower()
        if backend != "vibench":
            raise ValueError(f"Unsupported data backend: {backend!r}")

        code_root = str(self.cfg.get("vibench_code_root") or _default_vibench_code_root())
        _ensure_vibench_on_syspath(code_root)
        _ensure_pytorch_lightning_stub()

        build_data = None
        try:
            # NOTE:
            # PHMGA itself is also a top-level package named `src/`.
            # Importing PHM-Vibench's `src.data_factory` inside the same interpreter
            # can conflict with PHMGA's `src` module once it is imported/cached.
            #
            # We attempt the canonical vibench import first; if it fails, we fall back
            # to a lightweight loader that uses vibench `reader/` modules + metadata.
            from src.data_factory import build_data as _build_data  # type: ignore

            build_data = _build_data
        except Exception:
            build_data = None

        data_dir = Path(str(self.cfg["data_dir"]))
        data_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = str(self.cfg.get("metadata_file") or "metadata.xlsx")
        metadata_path = Path(metadata_file)
        if not metadata_path.is_absolute():
            metadata_path = data_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata_file not found: {str(metadata_path)!r}")

        dataset_name = str(self.cfg.get("dataset_name") or "").strip()
        rel_meta = ""
        if dataset_name:
            rel_meta = _write_filtered_metadata(
                data_dir=data_dir, metadata_path=metadata_path, dataset_name=dataset_name
            )
        else:
            rel_meta = (
                str(metadata_path.relative_to(data_dir))
                if str(metadata_path).startswith(str(data_dir))
                else str(metadata_path.name)
            )

        task_type = str(self.cfg.get("task_type") or "DG")
        task_name = str(self.cfg.get("task_name") or "Classification")

        args_data = SimpleNamespace(
            factory_name=str(self.cfg.get("factory_name") or "default"),
            data_dir=str(data_dir),
            metadata_file=str(rel_meta),
            batch_size=int(self.cfg.get("batch_size") or 32),
            num_workers=int(self.cfg.get("num_workers") or 0),
            pin_memory=bool(self.cfg.get("pin_memory") or False),
            window_size=int(self.cfg.get("window_size") or 4096),
            stride=int(self.cfg.get("stride") or 512),
            train_ratio=float(self.cfg.get("train_ratio") or 0.8),
            num_window=int(self.cfg.get("num_window") or 8),
            window_sampling_strategy=str(self.cfg.get("window_sampling_strategy") or "evenly_spaced"),
            normalization=str(self.cfg.get("normalization") or "standardization"),
            noise_snr=self.cfg.get("noise_snr"),
            dtype=str(self.cfg.get("dtype") or "float32"),
        )
        args_task = SimpleNamespace(
            type=task_type,
            name=task_name,
            target_system_id=self.cfg.get("target_system_id"),
            target_domain_num=int(self.cfg.get("target_domain_num") or 0),
            source_domain_id=self.cfg.get("source_domain_id"),
            target_domain_id=self.cfg.get("target_domain_id"),
        )

        # Label mapping from filtered metadata file.
        df = _read_metadata_table(data_dir / rel_meta)
        if "Label" not in df.columns:
            raise ValueError("Metadata file must contain a 'Label' column.")
        uniq = sorted({str(v) for v in df["Label"].dropna().unique().tolist()})
        if len(uniq) < 2:
            uniq = uniq + ["_dummy_1"]
        label_to_index = {lab: i for i, lab in enumerate(uniq)}

        # --- Path 1: Canonical PHM-Vibench factory ---
        if build_data is not None:
            factory = build_data(args_data, args_task)
            train_loader = factory.get_dataloader("train")
            val_loader = factory.get_dataloader("val")
            test_loader = factory.get_dataloader("test")

            wrapped = VibenchBuildResult(
                train_loader=_WrappedLoader(
                    train_loader, label_to_index=label_to_index, window_size=args_data.window_size
                ),
                val_loader=_WrappedLoader(
                    val_loader, label_to_index=label_to_index, window_size=args_data.window_size
                ),
                test_loader=_WrappedLoader(
                    test_loader, label_to_index=label_to_index, window_size=args_data.window_size
                ),
                label_to_index=label_to_index,
            )
            self._built = wrapped
            return wrapped

        # --- Path 2: Fallback loader (reader/ + metadata, avoids `src` name collision) ---
        try:
            import numpy as np  # type: ignore
            import torch
            from torch.utils.data import DataLoader, Dataset  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("torch/numpy are required for fallback vibench loader.") from e

        class _WindowDataset(Dataset):
            def __init__(self, items: List[Dict[str, Any]]):
                self._items = list(items)

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                return self._items[int(idx)]

        def _load_reader(name: str):
            p = Path(code_root) / "src" / "data_factory" / "reader" / f"{name}.py"
            if not p.exists():
                raise FileNotFoundError(f"vibench reader not found: {str(p)!r}")
            mod_name = f"phm_vibench_reader_{name}"
            spec = importlib.util.spec_from_file_location(mod_name, str(p))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load vibench reader: {name}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod

        def _evenly_spaced_windows(arr: "np.ndarray") -> List["np.ndarray"]:
            # arr: (L, C)
            L0 = int(arr.shape[0])
            ws = int(args_data.window_size)
            nw = int(getattr(args_data, "num_window", 8))
            if nw <= 0 or L0 < ws:
                return []
            if L0 == ws:
                return [arr.astype(np.float32, copy=False)]
            if nw == 1:
                start = (L0 - ws) // 2
                return [arr[start : start + ws].astype(np.float32, copy=False)]
            eff = L0 - ws
            step = eff / (nw - 1)
            out = []
            for i in range(nw):
                start = int(round(i * step))
                start = min(start, L0 - ws)
                out.append(arr[start : start + ws].astype(np.float32, copy=False))
            return out

        def _normalize(window: "np.ndarray") -> "np.ndarray":
            norm = str(getattr(args_data, "normalization", "standardization") or "standardization")
            if norm == "none":
                return window
            if norm in {"standardization", "z_score", "zscore"}:
                mu = window.mean(axis=0, keepdims=True)
                sd = window.std(axis=0, keepdims=True)
                return (window - mu) / (sd + 1e-8)
            if norm in {"minmax", "min_max"}:
                mn = window.min(axis=0, keepdims=True)
                mx = window.max(axis=0, keepdims=True)
                denom = mx - mn
                denom[denom == 0] = 1
                return (window - mn) / denom
            return window

        # Build per-id signals
        if "Id" not in df.columns or "Name" not in df.columns or "File" not in df.columns:
            raise ValueError("metadata must include columns: Id, Name, File")

        ids = [str(v) for v in df["Id"].tolist()]
        rng = np.random.default_rng(int(self.cfg.get("seed") or 42))
        perm = ids[:]
        rng.shuffle(perm)
        n = len(perm)
        n_train = max(1, int(round(float(args_data.train_ratio) * n))) if n else 0
        train_ids = perm[:n_train]
        rest = perm[n_train:]
        if not rest:
            rest = train_ids
        half = max(1, len(rest) // 2)
        val_ids = rest[:half]
        test_ids = rest[half:] if rest[half:] else val_ids

        id_to_row = {str(r["Id"]): r for _, r in df.iterrows()}

        def _build_items(split_ids: List[str]) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            for id_k in split_ids:
                row = id_to_row.get(str(id_k))
                if row is None:
                    continue
                name = str(row["Name"])
                file_name = str(row["File"])
                label = str(row.get("Label", "0"))
                mod = _load_reader(name)
                file_path = data_dir / "raw" / name / file_name
                arr = mod.read(str(file_path), args_data)  # type: ignore[attr-defined]
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr.reshape(arr.shape[0], -1)
                if arr.ndim != 2:
                    raise ValueError(f"Expected reader output (L,C), got {arr.shape} for id={id_k}")
                for w in _evenly_spaced_windows(arr):
                    w = _normalize(w)
                    items.append({"x": w, "y": label, "file_id": str(id_k)})
            return items

        def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            xs = [b["x"] for b in batch]
            ys = [b["y"] for b in batch]
            fids = [b.get("file_id") for b in batch]
            x = torch.as_tensor(np.stack(xs, axis=0), dtype=torch.float32)
            return {"x": x, "y": ys, "file_id": fids}

        train_ds = _WindowDataset(_build_items(train_ids))
        val_ds = _WindowDataset(_build_items(val_ids))
        test_ds = _WindowDataset(_build_items(test_ids))

        train_loader = DataLoader(train_ds, batch_size=int(args_data.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
        val_loader = DataLoader(val_ds, batch_size=int(args_data.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
        test_loader = DataLoader(test_ds, batch_size=int(args_data.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)

        wrapped = VibenchBuildResult(
            train_loader=_WrappedLoader(train_loader, label_to_index=label_to_index, window_size=args_data.window_size),
            val_loader=_WrappedLoader(val_loader, label_to_index=label_to_index, window_size=args_data.window_size),
            test_loader=_WrappedLoader(test_loader, label_to_index=label_to_index, window_size=args_data.window_size),
            label_to_index=label_to_index,
        )
        self._built = wrapped
        return wrapped
