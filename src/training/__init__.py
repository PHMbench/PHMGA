"""Training-runner exports for graph-path backends."""

from .runner import run_ml_pipeline, run_torch_pipeline

__all__ = [
    "GraphModule",
    "OperatorModuleFactory",
    "RuntimeNodeModule",
    "run_ml_pipeline",
    "run_torch_pipeline",
]


def __getattr__(name: str):
    if name in {"GraphModule", "OperatorModuleFactory", "RuntimeNodeModule"}:
        from .module_runtime import GraphModule, OperatorModuleFactory, RuntimeNodeModule

        return {
            "GraphModule": GraphModule,
            "OperatorModuleFactory": OperatorModuleFactory,
            "RuntimeNodeModule": RuntimeNodeModule,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
