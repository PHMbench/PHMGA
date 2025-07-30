import operator
from typing import TypedDict, Annotated

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import autosklearn.classification

from langgraph.graph import StateGraph, END

# 1. 定义图谱状态 (State)
# State 是图谱中各个节点之间传递的数据结构。
class AutoSklearnState(TypedDict):
    """
    表示 AutoML 工作流状态的 TypedDict。

    Attributes:
        x_train: 训练特征数据。
        y_train: 训练标签数据。
        x_test: 测试特征数据。
        y_test: 测试标签数据。
        model: 训练好的 AutoML 模型。
        score: 模型在测试集上的准确率得分。
        model_info: AutoML 找到的最佳模型的详细信息。
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    model: autosklearn.classification.AutoSklearnClassifier
    score: float
    model_info: str

# 2. 定义图谱节点 (Nodes)
# 每个节点都是一个函数，接收当前状态作为输入，并返回一个包含状态更新的字典。

def load_and_split_data(state: AutoSklearnState) -> dict:
    """
    加载数据并将其拆分为训练集和测试集。
    这里使用 scikit-learn 自带的 iris 数据集作为示例。
    """
    print("--- 节点: 加载并拆分数据 ---")
    X, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print("数据加载完成。")
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }

def train_automl_model(state: AutoSklearnState) -> dict:
    """
    使用 auto-sklearn 训练一个 AutoML 模型。
    """
    print("\n--- 节点: 训练 AutoML 模型 ---")
    x_train = state["x_train"]
    y_train = state["y_train"]

    # 初始化 AutoSklearnClassifier
    # time_left_for_this_task: 允许的总搜索时间（秒）
    # per_run_time_limit: 单次模型训练的限制时间（秒）
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=15,
        n_jobs=-1, # 使用所有可用的 CPU核心
    )
    
    print("AutoML 模型开始训练...")
    automl.fit(x_train, y_train)
    print("AutoML 模型训练完成。")

    # 获取找到的最佳模型的信息
    model_info = automl.show_models()
    print("找到的最佳模型信息:")
    print(model_info)

    return {"model": automl, "model_info": model_info}

def evaluate_model(state: AutoSklearnState) -> dict:
    """
    评估训练好的模型在测试集上的性能。
    """
    print("\n--- 节点: 评估模型 ---")
    model = state["model"]
    x_test = state["x_test"]
    y_test = state["y_test"]

    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(f"模型在测试集上的准确率: {score:.4f}")

    return {"score": score}

# 3. 构建图谱 (Graph)
workflow = StateGraph(AutoSklearnState)

# 添加节点
workflow.add_node("loader", load_and_split_data)
workflow.add_node("automl_trainer", train_automl_model)
workflow.add_node("evaluator", evaluate_model)

# 设置图谱的边 (Edges)
workflow.set_entry_point("loader")
workflow.add_edge("loader", "automl_trainer")
workflow.add_edge("automl_trainer", "evaluator")
workflow.add_edge("evaluator", END) # 评估后结束

# 4. 编译并运行图谱
app = workflow.compile()

print("=== 开始执行 AutoML 工作流 ===")
# 从一个空状态开始调用
final_state = app.invoke({})
print("\n=== AutoML 工作流执行完毕 ===")
print("\n最终状态:")
print(f"  - 测试集准确率: {final_state.get('score')}")
print(f"  - 找到的最佳模型数量: {len(final_state.get('model').cv_results_['params'])}")

# 你也可以直接运行这个脚本来查看效果
if __name__ == "__main__":
    # 为了方便直接运行，这里添加一个主函数入口
    print("\n\n=== 通过 main 函数直接运行 ===")
    runnable = workflow.compile()
    final_run_state = runnable.invoke({})
    
    print("\n=== 工作流执行完毕 ===")
    print("\n最终结果:")
    print(f"  - 测试集准确率: {final_run_state.get('score'):.4f}")
    print("\nAutoML 找到的最佳模型详情:")
    # show_models() 返回的是一个字符串，可以直接打印
    print(final_run_state.get('model_info'))
