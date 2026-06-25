#!/usr/bin/env python3
"""Generate responses from a local OpenAI-compatible API (e.g., vLLM) for IFBench evaluation."""

import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from config import get_settings


def load_prompts(input_file: str) -> list[dict]:
    """Load prompts from IFBench test file."""
    prompts = []
    with open(input_file, "r") as f:
        for line in f:
            example = json.loads(line)
            prompts.append({"key": example["key"], "prompt": example["prompt"]})
    return prompts

def generate_response(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
) -> tuple[str, str | None]:
    """Generate a response using the openai python client."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        #"extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "extra_body": {
            "top_k": top_k, 
            "min_p": min_p, 
            "repetition_penalty": repetition_penalty
        },
    }
    if seed is not None:
        kwargs["seed"] = seed

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    
    # 提取正文内容
    content = msg.content or ""
    content = content.strip()
    # 提取 reasoning 内容（vLLM 扩展字段）
    reasoning = getattr(msg, 'reasoning', None)
    if reasoning:
        reasoning = reasoning.strip()
    
    if "</think>" in content:
        # 按照 </think> 将文本切分为两部分
        parts = content.split("</think>", 1)
        
        # 提取 </think> 前的部分，并清理可能包含的 <think> 起始标签
        parsed_reasoning = parts[0].replace("<think>", "").strip()
        
        # 如果原生 msg.reasoning 已经有内容，则合并；否则直接赋值
        if reasoning:
            reasoning = reasoning + "\n\n" + parsed_reasoning
        else:
            reasoning = parsed_reasoning
            
        # 将 </think> 后的部分作为最终的正文内容
        content = parts[1].strip()
        
    return content, reasoning


def main():
    # Load settings from .env first
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Generate responses from a local OpenAI-compatible API for IFBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base",
        default=settings.api_base if hasattr(settings, 'api_base') else "http://localhost:8000/v1",
        help="Base URL for the OpenAI-compatible API",
    )

    parser.add_argument(
        "--model",
        default=settings.model if hasattr(settings, 'model') else "",
        help="Model name to use",
    )
    parser.add_argument(
        "--input-file",
        default=settings.input_file if hasattr(settings, 'input_file') else "../../datasets/data/ifeval/input_data.jsonl",
        help="Path to IFEval input file",
    )
    parser.add_argument(
        "--output-file",
        help="Output file for responses (defaults to data/{model}-responses.jsonl)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings.temperature if hasattr(settings, 'temperature') else 1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=settings.top_p if hasattr(settings, 'top_p') else 0.95,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--top-k",
        type=float,
        default=settings.top_k if hasattr(settings, 'top_k') else 20,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=settings.min_p if hasattr(settings, 'min_p') else 0.0,
        help="Min-p sampling",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=settings.presence_penalty if hasattr(settings, 'presence_penalty') else 1.5,
        help="Presence penalty",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=settings.repetition_penalty if hasattr(settings, 'repetition_penalty') else 1.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.max_tokens if hasattr(settings, 'max_tokens') else 4096,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.seed if hasattr(settings, 'seed') else None,
        help="Random seed for reproducibility (omit for random)",
    )
    parser.add_argument(
        "--api-key",
        default=settings.api_key if hasattr(settings, 'api_key') else "EMPTY",
        help="API key (defaults to 'EMPTY' for local models)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers if hasattr(settings, 'workers') else 4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )

    args = parser.parse_args()

    # Validate required settings
    if not args.model:
        parser.error("--model is required")
    if not args.api_base:
        parser.error("--api-base is required")

    prompts = load_prompts(args.input_file)
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    if not args.output_file:
        # 处理模型路径，防止包含 '/' 导致文件路径创建错误
        safe_model_name = args.model.strip('/').split('/')[-1]
        args.output_file = f"data/{safe_model_name}-responses.jsonl"
        
    print(f"Model: {args.model}")
    print(f"API: {args.api_base}")

    # Load existing responses if resuming
    existing_prompts = set()
    existing_responses = []
    if args.resume and Path(args.output_file).exists():
        with open(args.output_file, "r") as f:
            for line in f:
                resp = json.loads(line)
                existing_prompts.add(resp["prompt"])
                existing_responses.append(resp)
        print(f"Resuming: {len(existing_prompts)} prompts already completed")

    # Filter out completed prompts
    remaining = [p for p in prompts if p["prompt"] not in existing_prompts]
    print(f"Generating responses for {len(remaining)} prompts...")

    # Initialize OpenAI client
    # 默认本地 API_KEY 为 "EMPTY"
    client = OpenAI(base_url=args.api_base, api_key=args.api_key or "EMPTY")

    # Generate responses
    results = list(existing_responses)
    errors = []

    # OpenAI client 是线程安全的，可以直接传入多线程池
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_prompt = {
            executor.submit(
                generate_response,
                client,
                args.model,
                p["prompt"],
                args.temperature,
                args.max_tokens,
                args.seed,
                args.top_p,            
                args.top_k,            
                args.min_p,           
                args.presence_penalty, 
                args.repetition_penalty
            ): p
            for p in remaining
        }

        with tqdm(total=len(remaining), desc="Generating") as pbar:
            for future in as_completed(future_to_prompt):
                prompt_data = future_to_prompt[future]
                try:
                    content, reasoning = future.result()
                    results.append({
                        "prompt": prompt_data["prompt"],
                        "response": content,
                        "reasoning": reasoning,  # 将 reasoning 一并保存到 JSON 文件中
                    })
                except Exception as e:
                    errors.append({
                        "key": prompt_data["key"],
                        "error": str(e),
                    })
                    # Add empty response so eval can still run
                    results.append({
                        "prompt": prompt_data["prompt"],
                        "response": "",
                        "reasoning": "",
                    })
                pbar.update(1)

                # Save incrementally
                if len(results) % 10 == 0:
                    # 确保目标文件夹存在
                    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(args.output_file, "w") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final save
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} responses to {args.output_file}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  - Key {e['key']}: {e['error']}")

    print(f"\nRun evaluation with:")
    print(f"  python -m run_evaluation --input_data={args.input_file} --input_response_data={args.output_file} --output_dir=outputs/eval")


if __name__ == "__main__":
    main()
