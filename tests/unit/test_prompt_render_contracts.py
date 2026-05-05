from __future__ import annotations

from src.prompts.execute_prompt import render_execute_prompt
from src.prompts.plan_prompt import render_plan_prompt, render_supervisor_proving_plan_prompt
from src.prompts.reflect_prompt import render_reflect_prompt
from src.prompts.report_prompt import render_report_prompt


def test_plan_prompt_preserves_strict_json_contract_and_phm_guidance():
    prompt = render_plan_prompt(
        instruction="Build a richer PHM diagnosis DAG.",
        signal_context={"root_node_ids": ["ch1", "ch2"]},
        dag_json={},
        tools=[{"op_name": "fft", "schema_category": "TRANSFORM", "rank_class": "signal", "description": "fft", "planning_notes": "frequency"}],
        reflection=[],
        current_depth=0,
        min_depth=2,
        min_width=1,
    )
    assert "Return strict JSON only" in prompt
    assert "Time-domain analysis" in prompt
    assert "Envelope analysis" in prompt
    assert "You may branch from any legal existing node" in prompt
    assert "Do not reference nodes that would be created earlier in the same JSON response" in prompt
    assert "the DAG must end with executable aggregate feature outputs" in prompt


def test_supervisor_proving_prompt_stays_narrow_and_contract_strict():
    prompt = render_supervisor_proving_plan_prompt(
        instruction="Prove the workflow works.",
        signal_context={"root_node_ids": ["ch1"]},
        dag_json={},
        tools=[{"op_name": "fft", "schema_category": "TRANSFORM", "rank_class": "signal", "description": "fft", "planning_notes": "frequency"}],
        reflection=[],
        current_depth=0,
        min_depth=1,
        min_width=1,
    )
    assert "Return strict JSON only" in prompt
    assert "Do not use cross-channel, multi-parent, decision" in prompt
    assert "This proving lane exists to validate the workflow contract" in prompt


def test_reflect_prompt_preserves_current_decision_contract_and_architecture_language():
    prompt = render_reflect_prompt(
        instruction="Review the DAG.",
        stage="POST_EXECUTE",
        dag_blueprint={"nodes": []},
        dag_quality_summary={"issues": []},
        issues_summary="none",
        min_depth=2,
        min_width=1,
        max_depth=8,
        current_depth=2,
    )
    assert "Allowed decisions: finish, need_patch, need_replan, halt" in prompt
    assert "Evaluate operator diversity" in prompt
    assert "Evaluate symmetry across channels" in prompt
    assert "Treat `dag_quality_summary.dataset_level` as stronger evidence" in prompt


def test_report_prompt_stays_bound_to_current_artifact_inputs():
    prompt = render_report_prompt(
        instruction="Write the final report.",
        graph_path="ml",
        compiled_manifest={"graph_path": "ml"},
        path_artifacts={"metrics": {"test": {"accuracy": 0.8}}},
        reflection_summary={"decision": "finish"},
        dag_quality_summary={"issues": []},
        review_context={"note": "ok"},
    )
    assert "Graph path: ml" in prompt
    assert "compiled manifest and path artifacts as the primary factual source of truth" in prompt
    assert "Do not claim a richer backend or training stack than the artifacts support" in prompt


def test_execute_prompt_keeps_step_plan_materialization_contract():
    prompt = render_execute_prompt(
        step_plan={"plan": [{"parent": "ch1", "op_name": "fft", "params": {}}]},
        dag_json={"nodes": []},
        operator_catalog=[{"op_name": "fft"}],
        signal_context={"root_node_ids": ["ch1"]},
        graph_path="ml",
    )
    assert "Treat the plan as the execution source of truth" in prompt
    assert "emit an execution gap instead of inventing a fallback branch" in prompt
