# vLLM Architecture 核心抽象

## 两层抽象

### 1. LLMEngine - 核心推理引擎
负责模型加载、请求调度、KV cache管理

```python
from vllm import LLM, SamplingParams

# 初始化 - 加载模型到GPU
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

# 推理
prompts = ["What is AI?", "Explain transformers"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

**内部流程：**
```
LLM.__init__
  ├─> ModelRegistry.load_model()        # 加载Hugging Face模型
  ├─> PagedAttention初始化              # 核心优化：分页KV cache
  └─> CUDAGraph编译                     # kernel fusion优化

LLM.generate()
  ├─> Scheduler.add_request()           # 请求入队
  ├─> continuous batching               # 动态batching
  ├─> PagedAttention.forward()          # 高效注意力计算
  └─> TokenSampler.sample()             # 解码采样
```

### 2. AsyncLLMEngine - API Server层
基于LLMEngine，提供异步API接口

```python
# vLLM内部的API server实现
from vllm.entrypoints.openai.api_server import run_server

# 启动命令: vllm serve <model>
# 等价于:
engine = AsyncLLMEngine.from_engine_args(...)
app = FastAPI()

@app.post("/v1/completions")
async def completions(request):
    async for output in engine.generate(...):
        yield output  # streaming response
```

---

## 接入自定义模型的3种方式

### 方式1: Hugging Face兼容格式 (最简单)
只要你的模型是HF格式，vLLM自动识别

```
your-model/
├── config.json              # 模型配置
├── tokenizer.json           # 分词器
├── pytorch_model.bin        # 权重
└── generation_config.json
```

```python
# 直接用
llm = LLM(model="/path/to/your-model")
```

vLLM通过`config.json`里的`architectures`字段识别模型类型：
```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2"
}
```

### 方式2: 注册自定义模型类 (需要改vLLM源码)
如果你的模型架构vLLM不支持

```python
# vllm/model_executor/models/your_model.py
from vllm.model_executor.models import ModelRegistry

class YourCustomModel(nn.Module):
    def __init__(self, config):
        # 初始化权重
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],  # vLLM的PagedAttention cache
        input_metadata: InputMetadata,
    ):
        # 推理逻辑
        # 必须使用vLLM的PagedAttention算子
        hidden_states = self.embed(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                kv_cache=kv_caches[layer_idx],  # 关键：用vLLM的cache
                attn_metadata=input_metadata
            )
        return self.lm_head(hidden_states)

# 注册
ModelRegistry.register_model("YourModelType", YourCustomModel)
```

**关键点：**
- 必须用vLLM的`PagedAttention`算子，不能用原生PyTorch attention
- KV cache由vLLM的`CacheEngine`管理，不是自己管理

### 方式3: 只用vLLM的serving层，自己实现推理 (你的项目推荐)
**这是你做项目要走的路线：**

```python
# 不用vLLM的LLMEngine，只模仿它的接口
# 自己实现调度、batching、KV cache

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CustomInferenceEngine:
    def __init__(self, model_path: str):
        # 用原生PyTorch加载
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 你自己的KV cache管理
        self.kv_cache_manager = YourKVCacheManager()

        # 你自己的调度器
        self.scheduler = YourScheduler()

    async def generate(self, prompt: str, max_tokens: int):
        # 你的推理逻辑
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # 自己管理生成循环
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    past_key_values=self.kv_cache_manager.get(),  # 你的cache
                    use_cache=True
                )

            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # 更新cache
            self.kv_cache_manager.update(outputs.past_key_values)

            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids[0])

# API server
app = FastAPI()
engine = CustomInferenceEngine("/path/to/model")

@app.post("/generate")
async def generate(prompt: str):
    result = await engine.generate(prompt, max_tokens=100)
    return {"text": result}
```

---

## 你的项目应该怎么做

**第一阶段 - 理解baseline：**
1. 先跑通vLLM的API server，用benchmark测性能
2. 读vLLM源码，理解PagedAttention、continuous batching怎么实现的

**第二阶段 - 自己实现简化版：**
1. 用原生PyTorch + HuggingFace加载模型
2. 实现基础的KV cache管理（先不做paged，就是简单的tensor cache）
3. 实现request batching（先static batching，再做continuous）
4. 用FastAPI暴露API

**第三阶段 - 优化：**
1. 实现PagedAttention的简化版
2. 优化调度算法
3. benchmark对比你的实现 vs vLLM

**核心要学的：**
- `transformers`的`generate()`内部怎么做的 → 自动回归生成循环
- `past_key_values`怎么传递 → KV cache复用
- 怎么batch多个请求 → dynamic shapes处理

需要我写一个最简单的PyTorch原生推理demo吗？从这个开始，你就能理解怎么接入模型了。
