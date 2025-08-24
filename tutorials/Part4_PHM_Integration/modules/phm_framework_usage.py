"""
PHM框架使用模块

本模块展示如何使用项目中的实际PHM组件来构建完整的诊断系统。
包括DataAnalystAgent、PlanAgent等真实组件的集成使用。
"""

import sys
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

# 确保能导入项目模块
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.agents.data_analyst_agent import DataAnalystAgent
    from src.agents.plan_agent import PlanAgent  
    from src.states.phm_states import PHMState, DAGState, InputData
    from src.states.research_states import ResearchPHMState
    from src.model import get_llm
    PROJECT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 项目模块导入失败: {e}")
    print("将使用模拟组件进行演示")
    PROJECT_AVAILABLE = False


class PHMFrameworkDemo:
    """
    PHM框架使用演示
    
    展示如何集成和使用项目中的真实PHM组件
    """
    
    def __init__(self):
        self.setup_components()
    
    def setup_components(self):
        """设置PHM组件"""
        if PROJECT_AVAILABLE:
            try:
                # 使用真实的项目组件
                self.data_analyst = DataAnalystAgent(config={"quick_mode": False})
                self.plan_agent = PlanAgent()
                self.llm = get_llm(temperature=0.5)
                print("✅ 使用真实PHM组件")
            except Exception as e:
                print(f"⚠️ 真实组件初始化失败: {e}")
                self.setup_mock_components()
        else:
            self.setup_mock_components()
    
    def setup_mock_components(self):
        """设置模拟组件"""
        print("🔧 使用模拟PHM组件")
        
        class MockDataAnalyst:
            def __init__(self):
                self.agent_name = "MockDataAnalyst"
            
            def analyze(self, state):
                """模拟数据分析"""
                class MockResult:
                    def __init__(self):
                        self.confidence = 0.85
                        self.results = {
                            "frequency_analysis": "主频50Hz，存在2倍频谐波",
                            "amplitude_analysis": "振动幅值较高，超出正常范围15%",
                            "trend_analysis": "过去7天呈上升趋势"
                        }
                        self.execution_time = 2.3
                        self.memory_usage = 45.2
                        self.warnings = []
                        self.errors = []
                    
                    def is_successful(self):
                        return True
                
                return MockResult()
        
        class MockPlanAgent:
            def __init__(self):
                self.agent_name = "MockPlanAgent"
            
            def plan(self, state, analysis_result):
                """模拟维护计划"""
                plan = {
                    "immediate_actions": [
                        "停机检查轴承状态",
                        "测量轴承间隙",
                        "检查润滑系统"
                    ],
                    "short_term_plan": [
                        "更换磨损轴承",
                        "调整设备对中", 
                        "优化润滑方案"
                    ],
                    "long_term_strategy": [
                        "建立振动监控基线",
                        "制定预防性维护计划",
                        "培训操作人员"
                    ],
                    "estimated_cost": 15000,
                    "estimated_downtime": "4-6小时"
                }
                return plan
        
        self.data_analyst = MockDataAnalyst()
        self.plan_agent = MockPlanAgent()
    
    def create_bearing_fault_case(self) -> tuple:
        """创建轴承故障诊断案例"""
        # 生成模拟的轴承振动信号
        fs = 10000  # 采样频率
        t = np.linspace(0, 1, fs)
        
        # 健康轴承信号（主要是旋转频率）
        healthy_signal = (
            np.sin(2 * np.pi * 50 * t) +  # 50Hz转频
            0.1 * np.sin(2 * np.pi * 100 * t) +  # 2倍频
            0.05 * np.random.randn(len(t))  # 噪声
        )
        
        # 故障轴承信号（包含故障特征频率）
        faulty_signal = (
            np.sin(2 * np.pi * 50 * t) +  # 50Hz转频
            0.2 * np.sin(2 * np.pi * 100 * t) +  # 增强的2倍频
            0.3 * np.sin(2 * np.pi * 157 * t) +  # 轴承故障频率
            0.15 * np.sin(2 * np.pi * 314 * t) +  # 故障频率2倍频
            0.08 * np.random.randn(len(t))  # 噪声
        )
        
        # 调整信号形状以匹配期望的格式
        healthy_signal = healthy_signal.reshape(1, -1, 1)
        faulty_signal = faulty_signal.reshape(1, -1, 1)
        
        return healthy_signal, faulty_signal, fs
    
    def create_phm_state(self, reference_signal, test_signal, fs) -> Any:
        """创建PHM状态对象"""
        if PROJECT_AVAILABLE:
            try:
                # 使用真实的PHMState
                ref_data = InputData(
                    node_id="reference_bearing",
                    parents=[],
                    shape=reference_signal.shape,
                    results={"signal": reference_signal},
                    meta={"fs": fs, "sensor_type": "accelerometer"}
                )
                
                test_data = InputData(
                    node_id="test_bearing",
                    parents=[], 
                    shape=test_signal.shape,
                    results={"signal": test_signal},
                    meta={"fs": fs, "sensor_type": "accelerometer"}
                )
                
                dag_state = DAGState(
                    user_instruction="轴承故障诊断分析",
                    channels=["vibration_x"],
                    nodes={"ref": ref_data, "test": test_data},
                    leaves=["ref", "test"]
                )
                
                state = ResearchPHMState(
                    case_name="bearing_fault_diagnosis",
                    user_instruction="分析轴承振动信号，诊断可能的故障类型",
                    reference_signal=ref_data,
                    test_signal=test_data,
                    dag_state=dag_state,
                    fs=fs
                )
                
                return state
            except Exception as e:
                print(f"⚠️ 创建真实PHMState失败: {e}")
        
        # 使用模拟状态
        class MockPHMState:
            def __init__(self, ref_signal, test_signal, fs):
                self.case_name = "bearing_fault_diagnosis"
                self.user_instruction = "分析轴承振动信号，诊断可能的故障类型"
                self.reference_signal = {"data": ref_signal, "fs": fs}
                self.test_signal = {"data": test_signal, "fs": fs}
                self.fs = fs
        
        return MockPHMState(reference_signal, test_signal, fs)
    
    def run_complete_diagnosis(self) -> Dict[str, Any]:
        """运行完整的PHM诊断流程"""
        print("🏭 开始完整PHM诊断流程")
        print("=" * 50)
        
        # 1. 创建测试数据
        print("📊 生成轴承故障测试数据...")
        healthy_signal, faulty_signal, fs = self.create_bearing_fault_case()
        print(f"✅ 信号生成完成 (采样频率: {fs}Hz, 长度: {healthy_signal.shape[1]}点)")
        
        # 2. 创建PHM状态
        print("\\n🔧 创建PHM状态对象...")
        phm_state = self.create_phm_state(healthy_signal, faulty_signal, fs)
        print("✅ PHM状态创建完成")
        
        # 3. 数据分析
        print("\\n🔍 执行数据分析...")
        analysis_result = self.data_analyst.analyze(phm_state)
        print(f"✅ 数据分析完成 (置信度: {analysis_result.confidence:.1%})")
        print(f"   执行时间: {analysis_result.execution_time:.2f}秒")
        print(f"   内存使用: {analysis_result.memory_usage:.1f}MB")
        
        # 显示分析结果
        print("\\n📋 分析结果详情:")
        for key, value in analysis_result.results.items():
            print(f"   {key}: {value}")
        
        # 4. 生成维护计划
        print("\\n📝 生成维护计划...")
        maintenance_plan = self.plan_agent.plan(phm_state, analysis_result)
        print("✅ 维护计划生成完成")
        
        # 显示维护计划
        print("\\n🛠️ 维护计划详情:")
        if isinstance(maintenance_plan, dict):
            for category, actions in maintenance_plan.items():
                if isinstance(actions, list):
                    print(f"   {category}:")
                    for action in actions:
                        print(f"     - {action}")
                else:
                    print(f"   {category}: {actions}")
        else:
            print(f"   {maintenance_plan}")
        
        # 5. 综合结果
        diagnosis_result = {
            "case_info": {
                "case_name": getattr(phm_state, 'case_name', 'bearing_fault_diagnosis'),
                "signal_length": healthy_signal.shape[1],
                "sampling_frequency": fs
            },
            "analysis": {
                "agent": self.data_analyst.agent_name,
                "confidence": analysis_result.confidence,
                "results": analysis_result.results,
                "execution_time": analysis_result.execution_time,
                "success": analysis_result.is_successful()
            },
            "maintenance_plan": maintenance_plan,
            "overall_assessment": self._generate_overall_assessment(analysis_result, maintenance_plan)
        }
        
        print("\\n🎯 诊断流程完成!")
        return diagnosis_result
    
    def _generate_overall_assessment(self, analysis_result, maintenance_plan) -> Dict[str, Any]:
        """生成综合评估"""
        confidence = analysis_result.confidence
        
        if confidence > 0.8:
            risk_level = "高"
            urgency = "紧急"
            recommendation = "建议立即停机检修"
        elif confidence > 0.6:
            risk_level = "中等"
            urgency = "尽快"
            recommendation = "建议在下次停机时检修"
        else:
            risk_level = "低"
            urgency = "计划内"
            recommendation = "加强监控，按计划维护"
        
        return {
            "risk_level": risk_level,
            "urgency": urgency,
            "recommendation": recommendation,
            "confidence": confidence
        }
    
    def demonstrate_real_components(self):
        """演示真实组件的特殊功能"""
        if not PROJECT_AVAILABLE:
            print("⚠️ 真实组件不可用，跳过高级功能演示")
            return
        
        print("\\n🚀 真实PHM组件高级功能演示")
        print("=" * 50)
        
        try:
            # DataAnalyst的高级配置
            print("🔧 DataAnalyst高级配置:")
            advanced_config = {
                "quick_mode": False,
                "enable_advanced_features": True,
                "enable_pca_analysis": True,
                "use_parallel": True,
                "use_cache": True
            }
            
            advanced_analyst = DataAnalystAgent(config=advanced_config)
            print(f"   - 快速模式: {advanced_config['quick_mode']}")
            print(f"   - PCA分析: {advanced_config['enable_pca_analysis']}")
            print(f"   - 并行处理: {advanced_config['use_parallel']}")
            print(f"   - 结果缓存: {advanced_config['use_cache']}")
            
            # 性能监控
            performance_metrics = advanced_analyst.get_performance_metrics()
            if not performance_metrics.get("no_executions"):
                print("\\n📊 性能指标:")
                for key, value in performance_metrics.items():
                    print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"❌ 高级功能演示失败: {e}")


def run_comprehensive_demo():
    """运行综合演示"""
    print("🏭 PHM框架综合使用演示")
    print("=" * 60)
    
    # 创建演示实例
    demo = PHMFrameworkDemo()
    
    # 运行完整诊断
    result = demo.run_complete_diagnosis()
    
    # 显示最终总结
    print("\\n" + "=" * 60)
    print("📊 诊断总结报告")
    print("=" * 60)
    
    case_info = result["case_info"]
    analysis = result["analysis"]
    assessment = result["overall_assessment"]
    
    print(f"\\n📋 案例信息:")
    print(f"   案例名称: {case_info['case_name']}")
    print(f"   信号长度: {case_info['signal_length']:,} 点")
    print(f"   采样频率: {case_info['sampling_frequency']:,} Hz")
    
    print(f"\\n🔍 分析结果:")
    print(f"   分析Agent: {analysis['agent']}")
    print(f"   分析置信度: {analysis['confidence']:.1%}")
    print(f"   分析成功: {'是' if analysis['success'] else '否'}")
    print(f"   执行时间: {analysis['execution_time']:.2f}秒")
    
    print(f"\\n⚠️ 风险评估:")
    print(f"   风险等级: {assessment['risk_level']}")
    print(f"   紧急程度: {assessment['urgency']}")
    print(f"   建议: {assessment['recommendation']}")
    
    # 演示高级功能
    demo.demonstrate_real_components()
    
    return result


if __name__ == "__main__":
    run_comprehensive_demo()