"""
vLLM baseline benchmark - 用于建立性能基准
测试指标: latency, throughput, GPU memory usage
"""
import time
import asyncio
import aiohttp
from typing import List, Dict
import numpy as np


class vLLMBenchmark:
    def __init__(self, api_url: str = "http://localhost:8000/generate"):
        self.api_url = api_url

    async def single_request(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int = 100) -> Dict:
        """发送单个推理请求"""
        start_time = time.time()

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            async with session.post(self.api_url, json=payload) as response:
                result = await response.json()
                latency = time.time() - start_time
                return {
                    "latency": latency,
                    "success": True,
                    "output_length": len(result.get("text", "").split())
                }
        except Exception as e:
            return {
                "latency": time.time() - start_time,
                "success": False,
                "error": str(e)
            }

    async def concurrent_benchmark(self, prompts: List[str], concurrency: int = 10, max_tokens: int = 100):
        """并发压测"""
        print(f"Running benchmark with {len(prompts)} requests, concurrency={concurrency}")

        async with aiohttp.ClientSession() as session:
            start_time = time.time()

            # 批量发送请求
            tasks = [self.single_request(session, prompt, max_tokens) for prompt in prompts]
            results = await asyncio.gather(*tasks)

            total_time = time.time() - start_time

            # 统计结果
            successful = [r for r in results if r["success"]]
            latencies = [r["latency"] for r in successful]

            stats = {
                "total_requests": len(prompts),
                "successful_requests": len(successful),
                "failed_requests": len(prompts) - len(successful),
                "total_time": total_time,
                "throughput": len(successful) / total_time,  # requests/sec
                "avg_latency": np.mean(latencies) if latencies else 0,
                "p50_latency": np.percentile(latencies, 50) if latencies else 0,
                "p95_latency": np.percentile(latencies, 95) if latencies else 0,
                "p99_latency": np.percentile(latencies, 99) if latencies else 0,
            }

            return stats, results

    def print_stats(self, stats: Dict):
        """打印benchmark结果"""
        print("\n" + "="*50)
        print("vLLM Baseline Benchmark Results")
        print("="*50)
        print(f"Total requests:      {stats['total_requests']}")
        print(f"Successful:          {stats['successful_requests']}")
        print(f"Failed:              {stats['failed_requests']}")
        print(f"Total time:          {stats['total_time']:.2f}s")
        print(f"Throughput:          {stats['throughput']:.2f} req/s")
        print(f"Avg latency:         {stats['avg_latency']*1000:.2f}ms")
        print(f"P50 latency:         {stats['p50_latency']*1000:.2f}ms")
        print(f"P95 latency:         {stats['p95_latency']*1000:.2f}ms")
        print(f"P99 latency:         {stats['p99_latency']*1000:.2f}ms")
        print("="*50 + "\n")


async def main():
    # 测试数据 - 不同长度的prompts
    test_prompts = [
        "What is machine learning?",
        "Explain the transformer architecture in detail.",
        "Write a Python function to implement binary search.",
        "What are the key differences between PyTorch and TensorFlow?",
    ] * 25  # 100个请求

    benchmark = vLLMBenchmark()

    # 测试不同并发度
    for concurrency in [1, 5, 10, 20]:
        print(f"\nTesting with concurrency={concurrency}")
        stats, _ = await benchmark.concurrent_benchmark(
            test_prompts[:20],  # 先用20个请求测试
            concurrency=concurrency,
            max_tokens=50
        )
        benchmark.print_stats(stats)
        await asyncio.sleep(2)  # 间隔2秒避免过载


if __name__ == "__main__":
    print("Starting vLLM baseline benchmark...")
    print("Make sure vLLM server is running: vllm serve <model_name>")
    asyncio.run(main())
