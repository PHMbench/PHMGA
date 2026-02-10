from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.states.phm_states import DAGState, DAGTracker, InputData


def _make_tracker() -> DAGTracker:
    node = InputData(node_id="ch1", parents=[], shape=(1, 8, 1), data={}, results={})
    dag = DAGState(user_instruction="demo", channels=["ch1"], nodes={"ch1": node}, leaves=["ch1"])
    return DAGTracker(dag)


def test_write_png_fallback_writes_dot_no_empty_png(monkeypatch, tmp_path: Path):
    tracker = _make_tracker()

    def _raise_missing_graphviz(self):  # noqa: ANN001
        raise ModuleNotFoundError("No module named 'graphviz'")

    monkeypatch.setattr(DAGTracker, "to_dot", _raise_missing_graphviz)

    out_base = tmp_path / "graph_missing"
    ok = tracker.write_png(str(out_base))

    assert ok is False
    assert (tmp_path / "graph_missing.dot").exists()
    png = tmp_path / "graph_missing.png"
    assert (not png.exists()) or png.stat().st_size > 0


def test_write_png_success_returns_true(monkeypatch, tmp_path: Path):
    tracker = _make_tracker()

    class _FakeDot:
        source = "digraph G { ch1; }"

        def render(self, filename: str, format: str, cleanup: bool):  # noqa: A003
            assert format == "png"
            with open(f"{filename}.png", "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\nfake")
            return f"{filename}.png"

    monkeypatch.setattr(DAGTracker, "to_dot", lambda self: _FakeDot())

    out_base = tmp_path / "graph_ok"
    ok = tracker.write_png(str(out_base))

    assert ok is True
    png = tmp_path / "graph_ok.png"
    assert png.exists()
    assert png.stat().st_size > 0
