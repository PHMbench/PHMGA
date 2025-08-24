"""
Router模式模块

本模块展示不同的路由策略，包括：
- 基于规则的路由
- 基于LLM的智能路由
- 负载均衡路由
- 动态路由策略
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List
import time
import random


class AnalysisType(Enum):
    """分析类型枚举"""
    QUICK = "quick"
    STANDARD = "standard"
    DETAILED = "detailed"
    EMERGENCY = "emergency"


class BaseRouter(ABC):
    """路由器基类"""
    
    @abstractmethod
    def route(self, data: Dict[str, Any]) -> AnalysisType:
        """路由决策"""
        pass


class RuleBasedRouter(BaseRouter):
    """基于规则的路由器"""
    
    def __init__(self):
        # 定义阈值
        self.emergency_thresholds = {
            "temperature": 90,
            "vibration": 8.0,
            "pressure": 0.5
        }
        
        self.detailed_thresholds = {
            "temperature": 80,
            "vibration": 5.0,
            "pressure": 1.2
        }
    
    def route(self, data: Dict[str, Any]) -> AnalysisType:
        """基于规则的路由决策"""
        sensor_data = data.get("sensor_data", {})
        
        # 检查紧急情况
        for sensor, threshold in self.emergency_thresholds.items():
            value = sensor_data.get(sensor, 0)
            if (sensor == "pressure" and value < threshold) or \
               (sensor != "pressure" and value > threshold):
                return AnalysisType.EMERGENCY
        
        # 检查是否需要详细分析
        detailed_triggers = 0
        for sensor, threshold in self.detailed_thresholds.items():
            value = sensor_data.get(sensor, 0)
            if (sensor == "pressure" and value < threshold) or \
               (sensor != "pressure" and value > threshold):
                detailed_triggers += 1
        
        if detailed_triggers >= 2:
            return AnalysisType.DETAILED
        elif detailed_triggers == 1:
            return AnalysisType.STANDARD
        else:
            return AnalysisType.QUICK


class LLMBasedRouter(BaseRouter):
    """基于LLM的智能路由器"""
    
    def __init__(self, llm_provider: str = "mock"):
        self.llm_provider = llm_provider
        self.fallback_router = RuleBasedRouter()
    
    def route(self, data: Dict[str, Any]) -> AnalysisType:
        """基于LLM的智能路由"""
        try:
            # 模拟LLM调用（实际应该调用真实LLM）
            sensor_data = data.get("sensor_data", {})
            user_priority = data.get("priority", "normal")
            
            # 简化的LLM模拟逻辑
            if user_priority == "urgent":
                return AnalysisType.EMERGENCY
            
            # 基于传感器数据的简单启发式
            max_value = max(sensor_data.values()) if sensor_data else 0
            if max_value > 85:
                return AnalysisType.EMERGENCY
            elif max_value > 75:
                return AnalysisType.DETAILED
            elif max_value > 65:
                return AnalysisType.STANDARD
            else:
                return AnalysisType.QUICK
                
        except Exception as e:
            print(f"⚠️ LLM路由失败: {e}，使用规则回退")
            return self.fallback_router.route(data)


class LoadBalancingRouter(BaseRouter):
    """负载均衡路由器"""
    
    def __init__(self):
        self.request_count = 0
        self.analysis_loads = {
            AnalysisType.QUICK: 0,
            AnalysisType.STANDARD: 0,
            AnalysisType.DETAILED: 0,
            AnalysisType.EMERGENCY: 0
        }
        self.base_router = RuleBasedRouter()
    
    def route(self, data: Dict[str, Any]) -> AnalysisType:
        """负载均衡路由决策"""
        self.request_count += 1
        
        # 获取基础路由建议
        suggested_type = self.base_router.route(data)
        
        # 如果是紧急情况，直接路由
        if suggested_type == AnalysisType.EMERGENCY:
            self.analysis_loads[suggested_type] += 1
            return suggested_type
        
        # 检查负载均衡
        min_load_type = min(self.analysis_loads.keys(), key=lambda x: self.analysis_loads[x])
        
        # 如果建议类型的负载不是最高的，使用建议类型
        if self.analysis_loads[suggested_type] <= self.analysis_loads[min_load_type] + 2:
            final_type = suggested_type
        else:
            # 否则使用负载最低的类型（除了EMERGENCY）
            available_types = [t for t in self.analysis_loads.keys() if t != AnalysisType.EMERGENCY]
            final_type = min(available_types, key=lambda x: self.analysis_loads[x])
        
        self.analysis_loads[final_type] += 1
        return final_type
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """获取负载统计"""
        return {
            "total_requests": self.request_count,
            "load_distribution": {k.value: v for k, v in self.analysis_loads.items()}
        }


class AdaptiveRouter(BaseRouter):
    """自适应路由器"""
    
    def __init__(self):
        self.performance_history = {}
        self.router_weights = {
            "rule": 0.7,
            "llm": 0.3
        }
        self.rule_router = RuleBasedRouter()
        self.llm_router = LLMBasedRouter()
    
    def route(self, data: Dict[str, Any]) -> AnalysisType:
        """自适应路由决策"""
        # 获取两种路由的建议
        rule_suggestion = self.rule_router.route(data)
        llm_suggestion = self.llm_router.route(data)
        
        # 基于权重和历史性能决定最终路由
        if random.random() < self.router_weights["rule"]:
            final_decision = rule_suggestion
            used_router = "rule"
        else:
            final_decision = llm_suggestion
            used_router = "llm"
        
        # 记录决策
        self._record_decision(used_router, final_decision, data)
        
        return final_decision
    
    def _record_decision(self, router_type: str, decision: AnalysisType, data: Dict[str, Any]):
        """记录路由决策"""
        if router_type not in self.performance_history:
            self.performance_history[router_type] = []
        
        self.performance_history[router_type].append({
            "decision": decision,
            "timestamp": time.time(),
            "data_complexity": len(data.get("sensor_data", {}))
        })
    
    def update_weights(self, feedback: Dict[str, float]):
        """根据反馈更新路由权重"""
        total_score = sum(feedback.values())
        if total_score > 0:
            for router, score in feedback.items():
                if router in self.router_weights:
                    self.router_weights[router] = score / total_score
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            "current_weights": self.router_weights.copy(),
            "decision_history": {}
        }
        
        for router_type, history in self.performance_history.items():
            summary["decision_history"][router_type] = {
                "total_decisions": len(history),
                "recent_decisions": [h["decision"].value for h in history[-10:]]
            }
        
        return summary


class RouterDemo:
    """路由系统演示"""
    
    def __init__(self):
        self.routers = {
            "rule": RuleBasedRouter(),
            "llm": LLMBasedRouter(),
            "load_balancing": LoadBalancingRouter(),
            "adaptive": AdaptiveRouter()
        }
    
    def demo_routing_comparison(self):
        """演示路由策略比较"""
        test_cases = [
            {
                "name": "正常状态",
                "data": {
                    "sensor_data": {"temperature": 65, "vibration": 2.5, "pressure": 2.0},
                    "priority": "normal"
                }
            },
            {
                "name": "单项异常",
                "data": {
                    "sensor_data": {"temperature": 82, "vibration": 3.0, "pressure": 2.2},
                    "priority": "normal"
                }
            },
            {
                "name": "多项异常",
                "data": {
                    "sensor_data": {"temperature": 85, "vibration": 6.0, "pressure": 1.8},
                    "priority": "high"
                }
            },
            {
                "name": "紧急情况",
                "data": {
                    "sensor_data": {"temperature": 95, "vibration": 9.0, "pressure": 0.3},
                    "priority": "urgent"
                }
            }
        ]
        
        print("🔀 路由策略比较")
        print(f"{'场景':<15} {'规则路由':<15} {'LLM路由':<15} {'负载均衡':<15} {'自适应':<15}")
        print("-" * 80)
        
        for case in test_cases:
            results = {}
            for name, router in self.routers.items():
                results[name] = router.route(case["data"]).value
            
            print(f"{case['name']:<15} {results['rule']:<15} {results['llm']:<15} "
                  f"{results['load_balancing']:<15} {results['adaptive']:<15}")
        
        return results
    
    def demo_load_balancing(self):
        """演示负载均衡"""
        print("\n⚖️ 负载均衡演示")
        
        lb_router = self.routers["load_balancing"]
        
        # 模拟多个请求
        test_requests = [
            {"sensor_data": {"temperature": 70, "vibration": 3.0}},
            {"sensor_data": {"temperature": 85, "vibration": 6.0}},
            {"sensor_data": {"temperature": 60, "vibration": 2.0}},
            {"sensor_data": {"temperature": 90, "vibration": 8.0}},
            {"sensor_data": {"temperature": 75, "vibration": 4.0}},
        ] * 3  # 重复3次，总共15个请求
        
        for i, request in enumerate(test_requests):
            result = lb_router.route(request)
            if i % 5 == 4:  # 每5个请求输出一次统计
                stats = lb_router.get_load_statistics()
                print(f"请求 {i+1}: 负载分布 {stats['load_distribution']}")
        
        final_stats = lb_router.get_load_statistics()
        print(f"最终统计: {final_stats}")
        
        return final_stats
    
    def demo_adaptive_routing(self):
        """演示自适应路由"""
        print("\n🔄 自适应路由演示")
        
        adaptive_router = self.routers["adaptive"]
        
        # 初始权重
        print(f"初始权重: {adaptive_router.router_weights}")
        
        # 模拟一些决策
        test_data = [
            {"sensor_data": {"temperature": 70, "vibration": 3.0}},
            {"sensor_data": {"temperature": 85, "vibration": 6.0}},
            {"sensor_data": {"temperature": 75, "vibration": 4.0}}
        ]
        
        for data in test_data:
            result = adaptive_router.route(data)
            print(f"数据: {data['sensor_data']} -> 路由: {result.value}")
        
        # 模拟反馈更新权重
        feedback = {"rule": 0.6, "llm": 0.8}  # LLM表现更好
        adaptive_router.update_weights(feedback)
        
        print(f"反馈后权重: {adaptive_router.router_weights}")
        
        # 获取性能摘要
        summary = adaptive_router.get_performance_summary()
        print(f"性能摘要: {summary}")
        
        return summary


def demo_all_routers():
    """演示所有路由器"""
    print("🎯 Router模式综合演示")
    print("=" * 60)
    
    demo = RouterDemo()
    
    # 路由比较
    comparison_results = demo.demo_routing_comparison()
    
    # 负载均衡演示
    load_balance_results = demo.demo_load_balancing()
    
    # 自适应路由演示
    adaptive_results = demo.demo_adaptive_routing()
    
    return {
        "comparison": comparison_results,
        "load_balancing": load_balance_results,
        "adaptive": adaptive_results
    }


if __name__ == "__main__":
    demo_all_routers()