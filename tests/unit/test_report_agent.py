from __future__ import annotations

from src.agents import report_agent as report_module


class _DummyLLM:
    mode = "provider"

    def __init__(self):
        self.prompt = ""

    def generate_text(self, prompt: str) -> str:
        self.prompt = prompt
        return "# report"


def test_report_agent_omits_model_b64_from_prompt(monkeypatch):
    dummy = _DummyLLM()
    monkeypatch.setattr(report_module, "get_llm", lambda runtime_config=None: dummy)

    out = report_module.report_agent(
        instruction="diagnose bearing faults",
        dag_overview={"nodes": ["ch1"]},
        similarity_stats={"leaf": {"cosine": {"a": {"b": 0.1}}}},
        ml_results={
            "models": {
                "fft_01_ch1": {
                    "metrics": {"accuracy": 1.0},
                    "model_b64": "very-large-model-payload",
                }
            },
            "ensemble_metrics": {"accuracy": 1.0},
            "metrics_markdown": "| model | accuracy |\n|---|---|\n| fft_01_ch1 | 1.0 |",
        },
    )

    assert out["report_markdown"] == "# report"
    assert "very-large-model-payload" not in dummy.prompt
    assert '"accuracy": 1.0' in dummy.prompt


def test_report_agent_includes_node_level_results_and_final_selection(monkeypatch):
    dummy = _DummyLLM()
    monkeypatch.setattr(report_module, "get_llm", lambda runtime_config=None: dummy)

    out = report_module.report_agent(
        instruction="diagnose bearing faults",
        dag_overview={"nodes": ["ch1"]},
        similarity_stats={},
        ml_results={
            "models": {
                "leaf_a": {
                    "metrics": {"accuracy": 1.0, "macro_f1": 1.0},
                    "model_b64": "very-large-model-payload",
                }
            },
            "node_level_results": [
                {
                    "node_id": "leaf_a",
                    "feature_dim": 8,
                    "val_accuracy": 1.0,
                    "val_macro_f1": 1.0,
                    "test_accuracy": 1.0,
                    "test_macro_f1": 1.0,
                }
            ],
            "final_selection": {
                "best_single_leaf": "leaf_a",
                "final_choice": "best_single_leaf",
            },
            "ensemble_metrics": {"accuracy": 1.0, "macro_f1": 1.0},
            "metrics_markdown": "| model | accuracy |\n|---|---|\n| leaf_a | 1.0 |",
        },
    )

    assert out["report_markdown"] == "# report"
    assert "leaf_a" in dummy.prompt
    assert "best_single_leaf" in dummy.prompt
    assert "very-large-model-payload" not in dummy.prompt
