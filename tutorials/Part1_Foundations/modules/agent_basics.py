"""
Agent基础概念模块

本模块实现了从简单规则Agent到LLM增强Agent的演进过程，
帮助理解Agent的核心概念和发展历程。
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent基础抽象类"""
    
    def __init__(self, name: str):
        self.name = name
        self.memory: List[Dict[str, Any]] = []
        self.step_count = 0
        
    @abstractmethod
    def perceive(self, input_data: Any) -> Any:
        """感知环境"""
        pass
    
    @abstractmethod
    def think(self, perception: Any) -> Any:
        """思考和决策"""
        pass
    
    @abstractmethod
    def act(self, decision: Any) -> Any:
        """执行行动"""
        pass
    
    def run_cycle(self, input_data: Any) -> Any:
        """完整的Agent运行循环"""
        self.step_count += 1
        
        # 感知 -> 思考 -> 行动
        perception = self.perceive(input_data)
        decision = self.think(perception)
        action = self.act(decision)
        
        # 记录到记忆
        self.memory.append({
            "step": self.step_count,
            "input": input_data,
            "perception": perception,
            "decision": decision,
            "action": action,
            "timestamp": time.time()
        })
        
        return action


class SimpleRuleAgent(BaseAgent):
    """简单规则基础的Agent
    
    这是最基本的Agent实现，使用预定义的规则进行决策。
    适合用来理解Agent的基本概念和工作流程。
    """
    
    def __init__(self, name: str, rules: Optional[Dict[str, str]] = None):
        super().__init__(name)
        self.rules = rules or {
            "error": "fix_error",
            "warning": "investigate", 
            "normal": "continue",
            "unknown": "ask_for_help"
        }
    
    def perceive(self, input_data: str) -> Dict[str, Any]:
        """感知输入并提取关键信息"""
        perception = {
            "raw_input": input_data,
            "input_lower": input_data.lower(),
            "keywords": [],
            "severity": "normal"
        }
        
        # 关键词提取
        keywords = ["error", "warning", "fault", "alarm", "critical", "normal"]
        perception["keywords"] = [kw for kw in keywords if kw in perception["input_lower"]]
        
        # 严重程度判断
        if "error" in perception["input_lower"] or "critical" in perception["input_lower"]:
            perception["severity"] = "error"
        elif "warning" in perception["input_lower"] or "fault" in perception["input_lower"]:
            perception["severity"] = "warning"
        
        logger.info(f"🔍 {self.name} 感知到: {input_data} -> 严重程度: {perception['severity']}")
        return perception
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """基于规则进行决策"""
        severity = perception["severity"]
        action = self.rules.get(severity, self.rules["unknown"])
        
        decision = {
            "severity": severity,
            "action": action,
            "confidence": 0.8 if action != self.rules["unknown"] else 0.3,
            "reasoning": f"根据严重程度'{severity}'选择行动'{action}'"
        }
        
        logger.info(f"💭 {self.name} 决策: {decision['action']} (置信度: {decision['confidence']})")
        return decision
    
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行决策"""
        action_map = {
            "fix_error": "启动错误修复流程，通知维护团队",
            "investigate": "深入分析问题，收集更多数据", 
            "continue": "继续正常运行，保持监控",
            "ask_for_help": "请求人工介入，寻求专家意见"
        }
        
        action_description = action_map.get(decision["action"], "未知操作")
        
        result = {
            "action": decision["action"],
            "description": action_description,
            "status": "completed",
            "timestamp": time.time()
        }
        
        logger.info(f"🎬 {self.name} 执行: {action_description}")
        return result


class ReactiveAgent(BaseAgent):
    """反应式Agent
    
    能够根据环境变化做出快速响应的Agent。
    具有更复杂的感知和决策能力。
    """
    
    def __init__(self, name: str, response_threshold: float = 0.5):
        super().__init__(name)
        self.response_threshold = response_threshold
        self.environmental_state = {"temperature": 0.5, "pressure": 0.5, "vibration": 0.5}
    
    def perceive(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """感知多维环境数据"""
        # 更新环境状态
        for key, value in input_data.items():
            if key in self.environmental_state:
                self.environmental_state[key] = value
        
        # 计算异常程度
        anomaly_score = max(abs(v - 0.5) for v in self.environmental_state.values())
        
        perception = {
            "environmental_state": self.environmental_state.copy(),
            "anomaly_score": anomaly_score,
            "requires_action": anomaly_score > self.response_threshold,
            "most_abnormal": max(self.environmental_state.items(), key=lambda x: abs(x[1] - 0.5))
        }
        
        logger.info(f"🔍 {self.name} 环境感知: 异常分数={anomaly_score:.3f}")
        return perception
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """基于环境状态进行适应性决策"""
        if not perception["requires_action"]:
            decision = {
                "action": "monitor",
                "priority": "low",
                "reasoning": "系统状态正常，继续监控"
            }
        else:
            abnormal_param, abnormal_value = perception["most_abnormal"]
            
            if abnormal_value > 0.8:
                action, priority = "emergency_shutdown", "critical"
                reasoning = f"{abnormal_param}严重超标({abnormal_value:.3f})，需要紧急停机"
            elif abnormal_value > 0.7:
                action, priority = "reduce_load", "high"
                reasoning = f"{abnormal_param}超标({abnormal_value:.3f})，降低负载"
            else:
                action, priority = "adjust_parameters", "medium"
                reasoning = f"{abnormal_param}轻微异常({abnormal_value:.3f})，调整参数"
            
            decision = {
                "action": action,
                "priority": priority,
                "target_parameter": abnormal_param,
                "abnormal_value": abnormal_value,
                "reasoning": reasoning
            }
        
        logger.info(f"💭 {self.name} 决策: {decision['action']} ({decision['priority']}优先级)")
        return decision
    
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行适应性行动"""
        action_effects = {
            "monitor": lambda: "继续监控系统状态",
            "adjust_parameters": lambda: f"调整{decision.get('target_parameter', '参数')}",
            "reduce_load": lambda: f"降低{decision.get('target_parameter', '系统')}负载",
            "emergency_shutdown": lambda: "执行紧急停机程序"
        }
        
        action = decision["action"]
        effect = action_effects.get(action, lambda: "未知操作")()
        
        result = {
            "action": action,
            "effect": effect,
            "priority": decision["priority"],
            "success": True,
            "timestamp": time.time()
        }
        
        logger.info(f"🎬 {self.name} 执行: {effect}")
        return result


class LearningAgent(BaseAgent):
    """学习型Agent
    
    能够从经验中学习并改进决策的Agent。
    具有简单的学习和适应能力。
    """
    
    def __init__(self, name: str, learning_rate: float = 0.1):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.knowledge_base = {}
        self.success_history = []
    
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """感知并基于历史经验解释"""
        # 基础感知
        situation_key = self._extract_situation_key(input_data)
        
        perception = {
            "raw_input": input_data,
            "situation_key": situation_key,
            "historical_context": self.knowledge_base.get(situation_key, {}),
            "confidence": self._calculate_perception_confidence(situation_key)
        }
        
        logger.info(f"🔍 {self.name} 感知情况: {situation_key} (置信度: {perception['confidence']:.3f})")
        return perception
    
    def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """基于学习经验进行决策"""
        situation_key = perception["situation_key"]
        historical_context = perception["historical_context"]
        
        if historical_context and "best_action" in historical_context:
            # 使用学习到的最佳行动
            action = historical_context["best_action"]
            confidence = historical_context.get("success_rate", 0.5)
            reasoning = f"基于历史经验，{action}在类似情况下成功率{confidence:.1%}"
        else:
            # 探索性决策
            action = self._explore_action(perception)
            confidence = 0.3
            reasoning = "新情况，采用探索性策略"
        
        decision = {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "situation_key": situation_key,
            "is_exploration": "best_action" not in historical_context
        }
        
        logger.info(f"💭 {self.name} 决策: {action} (置信度: {confidence:.3f})")
        return decision
    
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行行动并学习结果"""
        action = decision["action"]
        
        # 模拟行动执行和结果
        success = self._simulate_action_result(action)
        
        result = {
            "action": action,
            "success": success,
            "timestamp": time.time()
        }
        
        # 学习更新
        self._update_knowledge(decision["situation_key"], action, success)
        
        logger.info(f"🎬 {self.name} 执行: {action} -> {'成功' if success else '失败'}")
        return result
    
    def _extract_situation_key(self, input_data: Dict[str, Any]) -> str:
        """提取情况特征键"""
        # 简化的特征提取
        if isinstance(input_data, dict) and "type" in input_data:
            return input_data["type"]
        else:
            return "general"
    
    def _calculate_perception_confidence(self, situation_key: str) -> float:
        """计算感知置信度"""
        if situation_key in self.knowledge_base:
            experience_count = self.knowledge_base[situation_key].get("count", 0)
            return min(1.0, 0.3 + experience_count * 0.1)
        return 0.3
    
    def _explore_action(self, perception: Dict[str, Any]) -> str:
        """探索性行动选择"""
        possible_actions = ["analyze", "wait", "intervene", "escalate"]
        return possible_actions[len(self.memory) % len(possible_actions)]
    
    def _simulate_action_result(self, action: str) -> bool:
        """模拟行动结果"""
        # 简单的成功概率模拟
        success_probs = {
            "analyze": 0.8,
            "wait": 0.6,
            "intervene": 0.7,
            "escalate": 0.9
        }
        import random
        return random.random() < success_probs.get(action, 0.5)
    
    def _update_knowledge(self, situation_key: str, action: str, success: bool):
        """更新知识库"""
        if situation_key not in self.knowledge_base:
            self.knowledge_base[situation_key] = {"actions": {}, "count": 0}
        
        kb_entry = self.knowledge_base[situation_key]
        kb_entry["count"] += 1
        
        if action not in kb_entry["actions"]:
            kb_entry["actions"][action] = {"attempts": 0, "successes": 0}
        
        action_stats = kb_entry["actions"][action]
        action_stats["attempts"] += 1
        if success:
            action_stats["successes"] += 1
        
        # 更新最佳行动
        best_action = max(
            kb_entry["actions"].items(),
            key=lambda x: x[1]["successes"] / max(x[1]["attempts"], 1)
        )
        
        kb_entry["best_action"] = best_action[0]
        kb_entry["success_rate"] = best_action[1]["successes"] / max(best_action[1]["attempts"], 1)


def demonstrate_agent_evolution():
    """演示Agent的演进过程"""
    print("🤖 Agent演进演示")
    print("=" * 50)
    
    # 1. 简单规则Agent
    print("\n1️⃣ 简单规则Agent:")
    rule_agent = SimpleRuleAgent("规则监控Agent")
    
    test_inputs = [
        "System running normally",
        "Warning: temperature high", 
        "Critical error in pump system",
        "Unknown sensor reading"
    ]
    
    for inp in test_inputs:
        result = rule_agent.run_cycle(inp)
        print(f"  输入: '{inp}' -> 行动: {result['action']}")
    
    # 2. 反应式Agent
    print("\n2️⃣ 反应式Agent:")
    reactive_agent = ReactiveAgent("环境反应Agent")
    
    env_inputs = [
        {"temperature": 0.3, "pressure": 0.4},
        {"temperature": 0.8, "vibration": 0.9},
        {"pressure": 0.95, "temperature": 0.7}
    ]
    
    for env in env_inputs:
        result = reactive_agent.run_cycle(env)
        print(f"  环境: {env} -> 行动: {result['action']}")
    
    # 3. 学习型Agent
    print("\n3️⃣ 学习型Agent:")
    learning_agent = LearningAgent("自适应学习Agent")
    
    learning_inputs = [
        {"type": "sensor_fault", "severity": 0.6},
        {"type": "sensor_fault", "severity": 0.8},
        {"type": "network_issue", "severity": 0.5}
    ]
    
    for inp in learning_inputs:
        result = learning_agent.run_cycle(inp)
        print(f"  学习: {inp} -> 行动: {result['action']} ({'成功' if result['success'] else '失败'})")
    
    print(f"\n📚 学习Agent知识库: {learning_agent.knowledge_base}")
    
    return rule_agent, reactive_agent, learning_agent


if __name__ == "__main__":
    demonstrate_agent_evolution()