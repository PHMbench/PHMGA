from __future__ import annotations

import abc
import functools
import logging
from typing import Any, ClassVar, Dict, Tuple, Literal, Type
import uuid
import numpy as np
from pydantic import BaseModel, Field

# 建议：可以从一个共享的工具模块导入
# from utils.shape import assert_shape

logger = logging.getLogger(__name__)

# ---------- 1. 注册表与装饰器 ---------- #
OP_REGISTRY: Dict[str, Type["PHMOperator"]] = {}

def register_op(cls: Type["PHMOperator"]) -> Type["PHMOperator"]:
    """
    类装饰器：在定义时自动将算子类注册到全局 OP_REGISTRY 字典中。
    """
    if not hasattr(cls, 'op_name'):
        raise TypeError(f"Operator {cls.__name__} must have a 'op_name' ClassVar.")
    
    op_name = cls.op_name
    if op_name in OP_REGISTRY:
        # If the operator already exists, simply return the existing class. This
        # allows modules to be executed multiple times (e.g., via "python -m")
        # without raising spurious duplicate registration errors.
        return OP_REGISTRY[op_name]

    OP_REGISTRY[op_name] = cls
    logger.debug(f"Registered operator: '{op_name}' -> {cls.__name__}")
    return cls
def get_operator(op_name: str) -> Type["PHMOperator"]:
    """
    根据算子名称获取注册的算子类。
    如果未找到，抛出 KeyError。
    """
    if op_name not in OP_REGISTRY:
        raise KeyError(f"Operator '{op_name}' is not registered.")
    return OP_REGISTRY[op_name]



# ---------- 2. 抽象基类 ---------- #
RankClass = Literal["EXPAND", "TRANSFORM", "AGGREGATE", "DECISION", "MultiVariable"]

class PHMOperator(BaseModel, abc.ABC):
    """
    所有 PHM 算子的共同抽象父类。
    它定义了统一的接口、生命周期钩子和与 LangGraph 的集成点。
    """
    node_id: str = Field(default_factory=lambda: f"op_{uuid.uuid4().hex[:8]}")
    op_name: ClassVar[str]
    rank_class: ClassVar[RankClass]
    description: ClassVar[str]
    input_spec: ClassVar[str]
    output_spec: ClassVar[str]
    parent: str | list[str] = Field(default=None, description="上游节点 ID 或 ID 列表，表示依赖的输入节点。")
    kind: Literal["op"] = "op"


    # 运行时状态，由钩子自动填充，便于调试和检查点
    in_shape: Tuple[int, ...] | None = Field(default=None, description="最近一次执行时的输入形状。")
    out_shape: Tuple[int, ...] | None = Field(default=None, description="最近一次执行时的输出形状。")
    params: Dict[str, Any] = Field(default_factory=dict, description="算子参数字典，包含所有可配置的参数。")

    class Config:
        extra = "forbid"  # 不允许未定义的字段
        arbitrary_types_allowed = True
        frozen = True  # 算子实例一旦创建即不可变，保证图执行的纯粹性

    # --- 对 LangGraph 公开的统一接口 --- #
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray | dict:
        """
        使算子实例成为可调用对象，可直接作为 LangGraph 节点。
        这个方法封装了执行的核心逻辑和生命周期钩子。
        """
        self._before_call(x)
        y = self.execute(x, **kwargs)
        self._after_call(y)
        return y

    @abc.abstractmethod
    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray | dict:
        """
        子类必须实现此方法，包含算子的核心算法。
        """
        raise NotImplementedError

    # --- 生命周期钩子 --- #
    def _before_call(self, x: np.ndarray) -> None:
        """执行前钩子：记录输入形状。"""
        # 使用 object.__setattr__ 来修改 frozen model 的字段
        object.__setattr__(self, 'in_shape', tuple(x.shape))

    def _after_call(self, y: np.ndarray | dict) -> None:
        """执行后钩子：记录输出形状并打印日志。"""
        shape = tuple(y.shape) if isinstance(y, np.ndarray) else None
        object.__setattr__(self, 'out_shape', shape)
        logger.info(
            "Executed %s(op_name='%s'): input_shape=%s -> output_shape=%s",
            self.__class__.__name__, self.op_name, self.in_shape, self.out_shape
        )


# ---------- 3. 四大功能抽象基类 ---------- #
class ExpandOp(PHMOperator):
    """升维算子的基类 (rank ↑)"""
    rank_class: ClassVar[RankClass] = "EXPAND"

class TransformOp(PHMOperator):
    """同维变换算子的基类 (rank =)"""
    rank_class: ClassVar[RankClass] = "TRANSFORM"

class AggregateOp(PHMOperator):
    """降维聚合算子的基类 (rank ↓)"""
    rank_class: ClassVar[RankClass] = "AGGREGATE"

class MultiVariableOp(PHMOperator):
    """拼接算子的基类，通常用于将多个输入沿特定轴拼接或拆分成若干个输出。"""
    rank_class: ClassVar[RankClass] = "MultiVariable"
    input_dict: Dict[str, np.ndarray] = Field(default_factory=dict, description="输入字典，键为输入名称，值为对应的 NumPy 数组。")
    output_dict: Dict[str, np.ndarray] = Field(default_factory=dict, description="输出字典，键为输出名称，值为对应的 NumPy 数组。")


class DecisionOp(PHMOperator):
    """决策算子的基类，通常输出字典而非数组。"""
    rank_class: ClassVar[RankClass] = "DECISION"

    # 覆盖父类钩子以处理字典输出
    def _after_call(self, y: dict) -> None:
        object.__setattr__(self, 'out_shape', None)  # 决策节点无输出形状
        logger.info(
            "Executed %s(op_name='%s'): input_shape=%s -> output=%s",
            self.__class__.__name__, self.op_name, self.in_shape, y,
        )


if __name__ == "__main__":
    print("--- Testing signal_processing_schemas.py ---")

    @register_op
    class IdentityOp(TransformOp):
        op_name: ClassVar[str] = "identity"
        description: ClassVar[str] = "Identity transform"
        input_spec: ClassVar[str] = "(B, L, C)"
        output_spec: ClassVar[str] = "(B, L, C)"

        def execute(self, x: np.ndarray, **_) -> np.ndarray:
            return x

    data = np.zeros((1, 10, 1))
    op = IdentityOp()
    out = op(data)
    assert np.all(out == data)
    assert get_operator("identity") is IdentityOp

    print("\n--- signal_processing_schemas.py tests passed! ---")
