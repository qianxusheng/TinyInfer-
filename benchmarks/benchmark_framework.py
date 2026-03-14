"""
统一的benchmark框架 - 对比三种实现
1. Naive PyTorch (baseline)
2. vLLM (industry standard)
3. Your custom engine (你的优化)
"""
import time
import asyncio
import aiohttp
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BenchmarkResult:
    """统一的性能指标"""
    engine_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    throughput: float  # requests/sec
    avg_latency: float  # seconds
    p50_latency: float
    p95_latency: float
    p99_latency: float
    avg_tokens_per_sec: float = 0.0  # generation speed
    memory_usage_mb: float = 0.0


class InferenceEngineInterface(ABC):
    """
    统一接口 - 所有引擎必须实现这个
    这样benchmark代码可以无差别测试任何实现
    """

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int) -> Dict:
        """
        生成接口
        Returns: {
            "text": str,
            "tokens_generated": int,
            "time": float
        }
        """
        pass

    @abstractmethod
    async def warmup(self):
        """预热 - 避免第一次请求慢"""
        pass

    @abstractmethod
    async def shutdown(self):
        """清理资源"""
        pass


# ============= 三种实现的Adapter =============

class NaivePyTorchEngine(InferenceEngineInterface):
    """Naive实现 - 直接用transformers，没有任何优化"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    async def warmup(self):
        """延迟加载模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading naive PyTorch model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        # 预热
        await self.generate("Hello", max_tokens=10)

    async def generate(self, prompt: str, max_tokens: int) -> Dict:
        import torch

        start = time.time()

        # 最简单的实现 - 没有batching，没有KV cache优化
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start

        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])

        return {
            "text": text,
            "tokens_generated": tokens_generated,
            "time": elapsed
        }

    async def shutdown(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


class VLLMEngine(InferenceEngineInterface):
    """vLLM实现 - 通过API调用"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = f"{api_url}/v1/completions"
        self.session = None

    async def warmup(self):
        """确保vLLM server已经启动"""
        self.session = aiohttp.ClientSession()

        # 测试连接
        try:
            await self.generate("test", max_tokens=5)
            print("vLLM server ready")
        except Exception as e:
            raise RuntimeError(f"vLLM server not accessible: {e}")

    async def generate(self, prompt: str, max_tokens: int) -> Dict:
        start = time.time()

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        async with self.session.post(self.api_url, json=payload) as response:
            result = await response.json()
            elapsed = time.time() - start

            # vLLM的response格式
            choice = result['choices'][0]
            text = choice['text']
            tokens_generated = choice.get('tokens', max_tokens)

            return {
                "text": text,
                "tokens_generated": tokens_generated,
                "time": elapsed
            }

    async def shutdown(self):
        if self.session:
            await self.session.close()


class CustomEngine(InferenceEngineInterface):
    """你的自定义引擎 - 通过API调用"""

    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = f"{api_url}/generate"
        self.session = None

    async def warmup(self):
        self.session = aiohttp.ClientSession()

        try:
            await self.generate("test", max_tokens=5)
            print("Custom engine ready")
        except Exception as e:
            raise RuntimeError(f"Custom engine not accessible: {e}")

    async def generate(self, prompt: str, max_tokens: int) -> Dict:
        start = time.time()

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        async with self.session.post(self.api_url, json=payload) as response:
            result = await response.json()
            elapsed = time.time() - start

            return {
                "text": result.get("text", ""),
                "tokens_generated": result.get("tokens_generated", max_tokens),
                "time": elapsed
            }

    async def shutdown(self):
        if self.session:
            await self.session.close()


# ============= Benchmark执行器 =============

class BenchmarkRunner:
    """运行benchmark并收集结果"""

    def __init__(self, engine: InferenceEngineInterface, engine_name: str):
        self.engine = engine
        self.engine_name = engine_name

    async def run_benchmark(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        concurrency: int = 1
    ) -> BenchmarkResult:
        """
        运行benchmark
        concurrency=1: 串行测试 (测latency)
        concurrency>1: 并发测试 (测throughput)
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {self.engine_name}")
        print(f"Requests: {len(prompts)}, Concurrency: {concurrency}, Max tokens: {max_tokens}")
        print(f"{'='*60}")

        await self.engine.warmup()

        start_time = time.time()

        # 控制并发度
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_generate(prompt):
            async with semaphore:
                try:
                    return await self.engine.generate(prompt, max_tokens)
                except Exception as e:
                    return {"error": str(e), "time": 0}

        # 并发执行
        results = await asyncio.gather(*[bounded_generate(p) for p in prompts])

        total_time = time.time() - start_time

        # 统计
        successful = [r for r in results if "error" not in r]
        failed = len(prompts) - len(successful)

        latencies = [r["time"] for r in successful]
        total_tokens = sum(r.get("tokens_generated", 0) for r in successful)

        return BenchmarkResult(
            engine_name=self.engine_name,
            total_requests=len(prompts),
            successful_requests=len(successful),
            failed_requests=failed,
            total_time=total_time,
            throughput=len(successful) / total_time if total_time > 0 else 0,
            avg_latency=np.mean(latencies) if latencies else 0,
            p50_latency=np.percentile(latencies, 50) if latencies else 0,
            p95_latency=np.percentile(latencies, 95) if latencies else 0,
            p99_latency=np.percentile(latencies, 99) if latencies else 0,
            avg_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        )

    async def cleanup(self):
        await self.engine.shutdown()


# ============= 结果可视化 =============

class BenchmarkVisualizer:
    """对比可视化"""

    @staticmethod
    def print_comparison(results: List[BenchmarkResult]):
        """打印对比表格"""
        print("\n" + "="*100)
        print(f"{'Engine':<20} {'Throughput':<15} {'Avg Latency':<15} {'P95 Latency':<15} {'Tokens/sec':<15}")
        print("="*100)

        for r in results:
            print(f"{r.engine_name:<20} "
                  f"{r.throughput:>10.2f} req/s  "
                  f"{r.avg_latency*1000:>10.2f} ms  "
                  f"{r.p95_latency*1000:>10.2f} ms  "
                  f"{r.avg_tokens_per_sec:>10.2f} tok/s")

        print("="*100)

        # 计算speedup
        if len(results) > 1:
            baseline = results[0]
            print(f"\nSpeedup vs {baseline.engine_name}:")
            for r in results[1:]:
                speedup = r.throughput / baseline.throughput if baseline.throughput > 0 else 0
                print(f"  {r.engine_name}: {speedup:.2f}x")

    @staticmethod
    def plot_comparison(results: List[BenchmarkResult], save_path: str = "benchmark_results.png"):
        """画对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        engines = [r.engine_name for r in results]

        # 1. Throughput对比
        throughputs = [r.throughput for r in results]
        axes[0, 0].bar(engines, throughputs, color=['red', 'blue', 'green'][:len(engines)])
        axes[0, 0].set_title('Throughput (requests/sec)')
        axes[0, 0].set_ylabel('req/s')

        # 2. Latency对比
        avg_latencies = [r.avg_latency * 1000 for r in results]
        p95_latencies = [r.p95_latency * 1000 for r in results]
        x = np.arange(len(engines))
        width = 0.35
        axes[0, 1].bar(x - width/2, avg_latencies, width, label='Avg', color='skyblue')
        axes[0, 1].bar(x + width/2, p95_latencies, width, label='P95', color='orange')
        axes[0, 1].set_title('Latency (ms)')
        axes[0, 1].set_ylabel('ms')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(engines)
        axes[0, 1].legend()

        # 3. Tokens/sec对比
        tokens_per_sec = [r.avg_tokens_per_sec for r in results]
        axes[1, 0].bar(engines, tokens_per_sec, color=['red', 'blue', 'green'][:len(engines)])
        axes[1, 0].set_title('Generation Speed (tokens/sec)')
        axes[1, 0].set_ylabel('tokens/s')

        # 4. Speedup对比
        if len(results) > 1:
            baseline_throughput = results[0].throughput
            speedups = [r.throughput / baseline_throughput if baseline_throughput > 0 else 0
                       for r in results]
            axes[1, 1].bar(engines, speedups, color=['red', 'blue', 'green'][:len(engines)])
            axes[1, 1].axhline(y=1.0, color='black', linestyle='--', label='Baseline')
            axes[1, 1].set_title('Speedup vs Baseline')
            axes[1, 1].set_ylabel('Speedup (x)')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()


# ============= 使用示例 =============

async def main():
    # 测试数据
    test_prompts = [
        "What is machine learning?",
        "Explain Python in one sentence.",
        "What is the capital of France?",
        "How does a transformer model work?",
    ] * 5  # 20个请求

    results = []

    # 1. Naive PyTorch (需要在本地运行)
    # naive_engine = NaivePyTorchEngine("Qwen/Qwen2.5-3B-Instruct")
    # naive_runner = BenchmarkRunner(naive_engine, "Naive PyTorch")
    # naive_result = await naive_runner.run_benchmark(test_prompts, max_tokens=50, concurrency=1)
    # results.append(naive_result)
    # await naive_runner.cleanup()

    # 2. vLLM (需要先启动: vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000)
    try:
        vllm_engine = VLLMEngine("http://localhost:8000")
        vllm_runner = BenchmarkRunner(vllm_engine, "vLLM")
        vllm_result = await vllm_runner.run_benchmark(test_prompts, max_tokens=50, concurrency=10)
        results.append(vllm_result)
        await vllm_runner.cleanup()
    except Exception as e:
        print(f"vLLM benchmark failed: {e}")

    # 3. Your custom engine (需要先启动你的server)
    try:
        custom_engine = CustomEngine("http://localhost:8001")
        custom_runner = BenchmarkRunner(custom_engine, "Custom Engine")
        custom_result = await custom_runner.run_benchmark(test_prompts, max_tokens=50, concurrency=10)
        results.append(custom_result)
        await custom_runner.cleanup()
    except Exception as e:
        print(f"Custom engine benchmark failed: {e}")

    # 结果对比
    if results:
        BenchmarkVisualizer.print_comparison(results)
        BenchmarkVisualizer.plot_comparison(results)


if __name__ == "__main__":
    asyncio.run(main())
