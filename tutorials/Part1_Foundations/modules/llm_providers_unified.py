"""
统一LLM Provider接口模块

本模块提供了统一的LLM Provider接口，支持多种大语言模型提供商，
包括Google Gemini、OpenAI GPT、通义千问、智谱GLM等。

使用LangChain的BaseChatModel接口确保一致性。
"""

from typing import Optional, Dict, Any, List
import os
import warnings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage


class UnifiedLLMProvider:
    """统一的LLM Provider管理器"""
    
    SUPPORTED_PROVIDERS = {
        "google": {
            "package": "langchain_google_genai",
            "class": "ChatGoogleGenerativeAI",
            "default_model": "gemini-pro",
            "api_key_env": "GEMINI_API_KEY",
            "models": ["gemini-pro", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        },
        "openai": {
            "package": "langchain_openai", 
            "class": "ChatOpenAI",
            "default_model": "gpt-3.5-turbo",
            "api_key_env": "OPENAI_API_KEY",
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
        },
        "tongyi": {
            "package": "langchain_community.llms",
            "class": "Tongyi", 
            "default_model": "qwen-turbo",
            "api_key_env": "DASHSCOPE_API_KEY",
            "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-longcontext"]
        },
        "glm": {
            "package": "langchain_community.chat_models",
            "class": "ChatZhipuAI",
            "default_model": "glm-4",
            "api_key_env": "ZHIPUAI_API_KEY", 
            "models": ["glm-4", "glm-4-air", "glm-4-flash", "glm-3-turbo"]
        },
        "mock": {
            "package": "langchain_community.chat_models",
            "class": "FakeListChatModel",
            "default_model": "mock-model",
            "api_key_env": None,
            "models": ["mock-model", "test-model"]
        }
    }
    
    @classmethod
    def create_llm(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> BaseChatModel:
        """
        创建LLM实例
        
        参数:
            provider: 提供商名称 (google, openai, tongyi, glm, mock)
            model: 模型名称（可选，使用默认模型）
            api_key: API密钥（可选，从环境变量获取）
            temperature: 温度参数
            **kwargs: 其他参数
            
        返回:
            BaseChatModel: LLM实例
        """
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"不支持的Provider: {provider}. "
                f"支持的Provider: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )
        
        config = cls.SUPPORTED_PROVIDERS[provider]
        
        # 确定模型名称
        if model is None:
            model = config["default_model"]
        elif model not in config["models"]:
            warnings.warn(f"模型 {model} 可能不被 {provider} 支持")
        
        # 获取API密钥
        if api_key is None and config["api_key_env"]:
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                raise ValueError(
                    f"需要API密钥。请设置环境变量 {config['api_key_env']} "
                    f"或直接传入api_key参数"
                )
        
        # 动态导入和创建实例
        try:
            if provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=temperature,
                    **kwargs
                )
            
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model,
                    openai_api_key=api_key,
                    temperature=temperature,
                    **kwargs
                )
            
            elif provider == "tongyi":
                from langchain_community.llms import Tongyi
                return Tongyi(
                    model=model,
                    dashscope_api_key=api_key,
                    temperature=temperature,
                    **kwargs
                )
            
            elif provider == "glm":
                from langchain_community.chat_models import ChatZhipuAI
                return ChatZhipuAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    **kwargs
                )
            
            elif provider == "mock":
                from langchain_community.chat_models import FakeListChatModel
                responses = kwargs.get("responses", [
                    "这是一个模拟响应，用于演示和测试。",
                    "PHM（Prognostics and Health Management）是预测性健康管理技术。",
                    "Graph Agent使用图结构来组织和执行复杂的工作流程。",
                    "LangChain提供了统一的LLM接口，支持多种语言模型。"
                ])
                return FakeListChatModel(responses=responses)
                
        except ImportError as e:
            raise ImportError(
                f"无法导入 {provider} 所需的包。请安装：\n"
                f"pip install {cls._get_install_command(provider)}"
            ) from e
    
    @classmethod
    def _get_install_command(cls, provider: str) -> str:
        """获取安装命令"""
        install_commands = {
            "google": "langchain-google-genai",
            "openai": "langchain-openai",
            "tongyi": "langchain-community dashscope", 
            "glm": "langchain-community zhipuai",
            "mock": "langchain-community"
        }
        return install_commands.get(provider, "langchain-community")
    
    @classmethod
    def list_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """列出所有可用的Provider"""
        available = {}
        
        for provider, config in cls.SUPPORTED_PROVIDERS.items():
            # 检查API密钥是否可用
            api_key_available = (
                config["api_key_env"] is None or 
                bool(os.getenv(config["api_key_env"]))
            )
            
            # 检查包是否已安装
            try:
                if provider == "google":
                    import langchain_google_genai
                elif provider == "openai":
                    import langchain_openai
                elif provider == "tongyi":
                    import dashscope
                elif provider == "glm":
                    import zhipuai
                elif provider == "mock":
                    import langchain_community
                package_available = True
            except ImportError:
                package_available = False
            
            available[provider] = {
                "models": config["models"],
                "default_model": config["default_model"],
                "api_key_required": config["api_key_env"] is not None,
                "api_key_available": api_key_available,
                "package_available": package_available,
                "ready": package_available and (not config["api_key_env"] or api_key_available)
            }
        
        return available
    
    @classmethod
    def get_best_available_provider(cls) -> str:
        """获取最佳可用的Provider"""
        available = cls.list_available_providers()
        
        # 优先级顺序
        priority_order = ["google", "openai", "tongyi", "glm", "mock"]
        
        for provider in priority_order:
            if available[provider]["ready"]:
                return provider
        
        # 如果没有可用的，返回mock
        return "mock"


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    便捷函数：创建LLM实例
    
    如果不指定provider，将自动选择最佳可用的provider。
    """
    if provider is None:
        provider = UnifiedLLMProvider.get_best_available_provider()
        print(f"🤖 自动选择Provider: {provider}")
    
    return UnifiedLLMProvider.create_llm(
        provider=provider,
        model=model,
        **kwargs
    )


class LLMBenchmark:
    """LLM Provider基准测试工具"""
    
    def __init__(self, providers: Optional[List[str]] = None):
        self.providers = providers or ["google", "openai", "tongyi", "glm", "mock"]
        self.test_queries = [
            "什么是PHM（预测性健康管理）？",
            "解释Graph Agent的工作原理",
            "LangChain的主要优势是什么？"
        ]
    
    def run_benchmark(self) -> Dict[str, Dict[str, Any]]:
        """运行基准测试"""
        results = {}
        
        for provider in self.providers:
            try:
                llm = create_llm(provider)
                provider_results = self._test_provider(provider, llm)
                results[provider] = provider_results
            except Exception as e:
                results[provider] = {
                    "status": "error",
                    "error": str(e),
                    "avg_response_time": None,
                    "avg_response_length": 0
                }
        
        return results
    
    def _test_provider(self, provider: str, llm: BaseChatModel) -> Dict[str, Any]:
        """测试单个Provider"""
        import time
        
        response_times = []
        response_lengths = []
        responses = []
        
        for query in self.test_queries:
            try:
                start_time = time.time()
                response = llm.invoke(query)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_content = response.content if hasattr(response, 'content') else str(response)
                response_length = len(response_content)
                
                response_times.append(response_time)
                response_lengths.append(response_length)
                responses.append(response_content[:100] + "..." if len(response_content) > 100 else response_content)
                
            except Exception as e:
                print(f"❌ {provider} 查询失败: {e}")
                continue
        
        if response_times:
            return {
                "status": "success",
                "avg_response_time": sum(response_times) / len(response_times),
                "avg_response_length": sum(response_lengths) / len(response_lengths),
                "sample_responses": responses,
                "test_count": len(response_times)
            }
        else:
            return {
                "status": "failed",
                "error": "所有测试查询都失败了",
                "avg_response_time": None,
                "avg_response_length": 0
            }


def demo_provider_comparison():
    """演示不同Provider的对比"""
    print("🚀 LLM Provider对比演示")
    print("=" * 50)
    
    # 列出可用的Provider
    available = UnifiedLLMProvider.list_available_providers()
    print("\n📋 Provider可用性:")
    for provider, info in available.items():
        status = "✅ 就绪" if info["ready"] else "❌ 不可用"
        reason = ""
        if not info["ready"]:
            if not info["package_available"]:
                reason = " (缺少包)"
            elif not info["api_key_available"]:
                reason = " (缺少API密钥)"
        
        print(f"  {provider}: {status}{reason}")
    
    # 运行基准测试
    print("\n🧪 运行基准测试...")
    benchmark = LLMBenchmark()
    results = benchmark.run_benchmark()
    
    print("\n📊 基准测试结果:")
    for provider, result in results.items():
        if result["status"] == "success":
            print(f"\n  🤖 {provider}:")
            print(f"    平均响应时间: {result['avg_response_time']:.2f}秒")
            print(f"    平均响应长度: {result['avg_response_length']:.0f}字符")
            print(f"    测试成功数: {result['test_count']}")
            if result.get("sample_responses"):
                print(f"    示例响应: {result['sample_responses'][0]}")
        else:
            print(f"\n  ❌ {provider}: {result.get('error', '测试失败')}")
    
    return results


def interactive_provider_test():
    """交互式Provider测试"""
    print("🎮 交互式Provider测试")
    print("=" * 30)
    
    available_providers = UnifiedLLMProvider.list_available_providers()
    ready_providers = [p for p, info in available_providers.items() if info["ready"]]
    
    if not ready_providers:
        print("❌ 没有可用的Provider。请配置API密钥或使用Mock Provider。")
        return
    
    print(f"可用的Provider: {', '.join(ready_providers)}")
    
    while True:
        try:
            provider = input(f"\n选择Provider ({'/'.join(ready_providers)}) 或 'quit' 退出: ").strip()
            
            if provider.lower() == 'quit':
                break
            
            if provider not in ready_providers:
                print(f"❌ 无效的Provider。请选择: {', '.join(ready_providers)}")
                continue
            
            # 创建LLM实例
            llm = create_llm(provider)
            print(f"✅ 已创建 {provider} LLM实例")
            
            # 交互式查询
            while True:
                query = input("\n输入查询内容 (或 'back' 返回Provider选择): ").strip()
                
                if query.lower() == 'back':
                    break
                
                if not query:
                    continue
                
                try:
                    print(f"🤖 {provider} 正在思考...")
                    response = llm.invoke(query)
                    content = response.content if hasattr(response, 'content') else str(response)
                    print(f"\n💬 回答:\n{content}\n")
                except Exception as e:
                    print(f"❌ 查询失败: {e}")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    # 运行演示
    demo_provider_comparison()
    
    # 可选：运行交互式测试
    # interactive_provider_test()