from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENTS_ROOT = ROOT / "src" / "agents"


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _get_function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _arg_names(fn: ast.FunctionDef) -> list[str]:
    args = [arg.arg for arg in fn.args.args]
    args.extend(arg.arg for arg in fn.args.kwonlyargs)
    return args


def test_agent_entrypoints_define_expected_primary_functions():
    expected = {
        AGENTS_ROOT / "plan_agent.py": ("plan_agent", ["state"]),
        AGENTS_ROOT / "dag_init_agent.py": ("dag_init_agent", ["state", "max_ops_per_channel", "temperature"]),
        AGENTS_ROOT / "execute_agent.py": ("execute_agent", ["state"]),
        AGENTS_ROOT / "reflect_agent.py": ("reflect_agent_node", ["state", "stage"]),
        AGENTS_ROOT / "report_agent.py": ("report_agent_node", ["state"]),
        AGENTS_ROOT / "dataset_preparer_agent.py": ("dataset_preparer_agent", ["state", "config"]),
        AGENTS_ROOT / "deep_model_train_agent.py": ("deep_model_train_agent", ["state", "config"]),
        AGENTS_ROOT / "inquirer_agent.py": ("inquirer_agent", ["state", "metrics"]),
        AGENTS_ROOT / "shallow_ml_agent.py": ("shallow_ml_agent", ["datasets", "algorithm", "ensemble_method", "cv_folds"]),
        AGENTS_ROOT / "tspn_bootstrap_agent.py": (
            "tspn_bootstrap_agent",
            ["state", "max_layers", "parallel_ops_per_layer", "out_channels", "scale", "features"],
        ),
    }
    for path, (fn_name, expected_args) in expected.items():
        tree = _parse_module(path)
        fn = _get_function_def(tree, fn_name)
        assert _arg_names(fn) == expected_args
        assert fn.returns is not None, f"{path}::{fn_name} should declare a return annotation"


def test_deep_research_agent_family_exposes_expected_nodes():
    tree = _parse_module(AGENTS_ROOT / "deep_research_agents.py")
    required = {
        "generate_query": ["state", "config"],
        "web_research": ["state", "config"],
        "reflection": ["state", "config"],
        "evaluate_research": ["state", "config"],
        "finalize_answer": ["state", "config"],
    }
    for fn_name, expected_args in required.items():
        fn = _get_function_def(tree, fn_name)
        assert _arg_names(fn) == expected_args


def test_agent_wrapper_modules_remain_thin_re_exports():
    wrappers = [
        AGENTS_ROOT / "builder" / "plan.py",
        AGENTS_ROOT / "executor" / "execute.py",
        AGENTS_ROOT / "report" / "final_report.py",
        AGENTS_ROOT / "train" / "deep_model.py",
    ]
    for path in wrappers:
        tree = _parse_module(path)
        non_doc_nodes = [node for node in tree.body if not isinstance(node, ast.Expr)]
        assert len(non_doc_nodes) == 2, f"{path} should stay a thin import + __all__ wrapper"
        assert isinstance(non_doc_nodes[0], ast.ImportFrom)
        assert isinstance(non_doc_nodes[1], ast.Assign)


def test_agents_shared_compat_stays_a_passthrough_helper():
    tree = _parse_module(AGENTS_ROOT / "shared" / "compat.py")
    fn = _get_function_def(tree, "forward_agent_call")
    body = [node for node in fn.body if not isinstance(node, ast.Expr)]
    assert len(body) == 1
    assert isinstance(body[0], ast.Return)


def test_agents_directory_does_not_contain_pycache_directories():
    pycaches = sorted(str(path) for path in AGENTS_ROOT.rglob("__pycache__"))
    assert not pycaches
