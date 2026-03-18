| 模型显示名                                 | API 里的 `model` 值                                |       上下文窗口 | 免费方式          | 适合场景                           | 说明                                                                  |
| ------------------------------------- | ----------------------------------------------- | ----------: | ------------- | ------------------------------ | ------------------------------------------------------------------- |
| StepFun Step 3.5 Flash (free)         | `stepfun/step-3.5-flash:free`                   |        256K | 指定 `:free` 变体 | 通用、推理、长上下文                     | 模型页标为 Free variant，价格为 $0/M。 ([OpenRouter][1])                      |
| Hunter Alpha                          | `openrouter/hunter-alpha`                       |       1.05M | 直接免费模型        | 长上下文、agent、多步任务                | 模型页价格为 $0/M，并提示 provider 会记录 prompts/completions。 ([OpenRouter][2]) |
| Arcee Trinity Large Preview (free)    | `arcee-ai/trinity-large-preview:free`           |        131K | 指定 `:free` 变体 | 创作、对话、agent 工具链                | 模型页标为 Free variant，价格为 $0/M。 ([OpenRouter][3])                      |
| NVIDIA Nemotron 3 Super (free)        | `nvidia/nemotron-3-super-120b-a12b:free`        |        262K | 指定 `:free` 变体 | 多 agent、长任务、推理                 | Free variant，$0/M。 ([OpenRouter][4])                                |
| Z.ai GLM 4.5 Air (free)               | `z-ai/glm-4.5-air:free`                         |        131K | 指定 `:free` 变体 | agent、工具调用、可控 reasoning        | 模型页说明支持 reasoning 开关。 ([OpenRouter][5])                             |
| NVIDIA Nemotron 3 Nano 30B A3B (free) | `nvidia/nemotron-3-nano-30b-a3b:free`           |        256K | 指定 `:free` 变体 | 轻量 agent、开发测试                  | Free variant，$0/M。 ([OpenRouter][6])                                |
| Arcee Trinity Mini (free)             | `arcee-ai/trinity-mini:free`                    |        131K | 指定 `:free` 变体 | 轻量推理、函数调用                      | Free variant，$0/M。 ([OpenRouter][7])                                |
| NVIDIA Nemotron Nano 12B 2 VL (free)  | `nvidia/nemotron-nano-12b-v2-vl:free`           |        128K | 指定 `:free` 变体 | 多模态、文档/图像理解                    | 这是视觉语言免费模型。 ([OpenRouter][8])                                       |
| NVIDIA Nemotron Nano 9B V2 (free)     | `nvidia/nemotron-nano-9b-v2:free`               |        128K | 指定 `:free` 变体 | 轻量通用问答/推理                      | Free variant，$0/M。 ([OpenRouter][9])                                |
| Qwen3 Coder 480B A35B (free)          | `qwen/qwen3-coder:free`                         |        262K | 指定 `:free` 变体 | 代码、agent coding、仓库长上下文         | 免费版在 free 集合页中列出。 ([OpenRouter][10])                                |
| Qwen3 Next 80B A3B Instruct (free)    | `qwen/qwen3-next-80b-a3b-instruct:free`         |        262K | 指定 `:free` 变体 | 通用 assistant、长上下文              | 免费版在 free 集合页中列出。 ([OpenRouter][11])                                |
| Meta Llama 3.3 70B Instruct (free)    | `meta-llama/llama-3.3-70b-instruct:free`        |        128K | 指定 `:free` 变体 | 通用、多语言                         | Free variant，$0/M。 ([OpenRouter][12])                               |
| OpenAI gpt-oss-120b (free)            | `openai/gpt-oss-120b:free`                      |        131K | 指定 `:free` 变体 | 高推理、tool use、structured output | Free variant，$0/M。 ([OpenRouter][13])                               |
| Mistral Small 3.1 24B (free)          | `mistralai/mistral-small-3.1-24b-instruct:free` | 集合页列为免费热门模型 | 指定 `:free` 变体 | 轻量通用、代码、视觉                     | 免费热门模型之一。 ([OpenRouter][14])                                        |

[1]: https://openrouter.ai/stepfun/step-3.5-flash%3Afree "Step 3.5 Flash (free) - API Pricing & Providers | OpenRouter"
[2]: https://openrouter.ai/openrouter/hunter-alpha "Hunter Alpha - API Pricing & Providers | OpenRouter"
[3]: https://openrouter.ai/arcee-ai/trinity-large-preview%3Afree "Trinity Large Preview (free) - API Pricing & Providers | OpenRouter"
[4]: https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b%3Afree "Nemotron 3 Super (free) - API Pricing & Providers | OpenRouter"
[5]: https://openrouter.ai/z-ai/glm-4.5-air%3Afree "GLM 4.5 Air (free) - API Pricing & Providers | OpenRouter"
[6]: https://openrouter.ai/nvidia/nemotron-3-nano-30b-a3b%3Afree "Nemotron 3 Nano 30B A3B (free) - API Pricing & Providers | OpenRouter"
[7]: https://openrouter.ai/arcee-ai/trinity-mini%3Afree "Trinity Mini (free) - API Pricing & Providers | OpenRouter"
[8]: https://openrouter.ai/nvidia/nemotron-nano-12b-v2-vl%3Afree "Nemotron Nano 12B 2 VL (free) - API Pricing & Providers | OpenRouter"
[9]: https://openrouter.ai/nvidia/nemotron-nano-9b-v2%3Afree "Nemotron Nano 9B V2 (free) - API Pricing & Providers | OpenRouter"
[10]: https://openrouter.ai/qwen/qwen3-coder%3Afree "Qwen3 Coder 480B A35B (free) - API Pricing & Providers | OpenRouter"
[11]: https://openrouter.ai/qwen/qwen3-next-80b-a3b-instruct%3Afree "Qwen3 Next 80B A3B Instruct (free) - API Pricing & Providers"
[12]: https://openrouter.ai/meta-llama/llama-3.3-70b-instruct%3Afree "Llama 3.3 70B Instruct (free) - API Pricing & Providers | OpenRouter"
[13]: https://openrouter.ai/openai/gpt-oss-120b%3Afree "gpt-oss-120b (free) - API Pricing & Providers | OpenRouter"
[14]: https://openrouter.ai/mistralai/mistral-small-3.1-24b-instruct%3Afree "Mistral Small 3.1 24B (free) - API Pricing & Providers | OpenRouter"
