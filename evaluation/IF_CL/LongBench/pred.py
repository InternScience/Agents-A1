import os, json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
import re
from openai import OpenAI
from transformers import AutoTokenizer

# 压制 transformers 关于序列长度的警告（vLLM 已支持 1M 上下文）
import warnings
warnings.filterwarnings('ignore', message='.*Token indices sequence length.*')

from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent

model_map = json.loads((BASE_DIR / 'config/model2path.json').read_text(encoding='utf-8'))
maxlen_map = json.loads((BASE_DIR / 'config/model2maxlen.json').read_text(encoding='utf-8'))

template_rag = (BASE_DIR / 'prompts/0shot_rag.txt').read_text(encoding='utf-8')
template_no_context = (BASE_DIR / 'prompts/0shot_no_context.txt').read_text(encoding='utf-8')
template_0shot = (BASE_DIR / 'prompts/0shot.txt').read_text(encoding='utf-8')
template_0shot_cot = (BASE_DIR / 'prompts/0shot_cot.txt').read_text(encoding='utf-8')
template_0shot_cot_ans = (BASE_DIR / 'prompts/0shot_cot_ans.txt').read_text(encoding='utf-8')

def truncate_prompt(prompt, tokenizer, max_tokens):
    """原始简单截断策略"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens//2] + input_ids[-max_tokens//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt

def query_llm(prompt, model, client, tokenizer, max_input_tokens, temperature=0.1, max_new_tokens=128, timeout=600):
    prompt = truncate_prompt(prompt, tokenizer, max_input_tokens)
    for attempt in range(5):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                timeout=timeout,
            )
            # if "qwen3" in model.lower():
                # Qwen3 推理配置（YaRN 等配置已在服务器启动时设置）
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": True},
                # 采样参数（Thinking mode for general tasks）
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
            }
            kwargs["extra_body"] = extra_body
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content or ''
            reasoning_content = getattr(completion.choices[0].message, 'reasoning', '') or ''
            # 用think标签包裹reasoning
            result = ''
            if reasoning_content:
                result += f'<think>\n{reasoning_content}\n</think>'
            if content:
                result += '\n' + content if result else content
            return result
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"[Attempt {attempt+1}/5] Error: {str(e)}")
            time.sleep(1)
    print("[query_llm] Max tries. Failed.")
    return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', response)
    if match:
        return match.group(1)
    return None

def process_single(item, args, client, tokenizer, max_input_tokens):
    item_id = item["_id"]
    try:
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())

        # Qwen3 使用 temperature=1.0（推理模式推荐），其他模型使用 0.1
        temp = 1.0 if "qwen3" in args.model.lower() else 1.0

        if args.cot:
            output = query_llm(prompt, args.model, client, tokenizer, max_input_tokens, temperature=temp, max_new_tokens=1024, timeout=args.timeout)
        else:
            output = query_llm(prompt, args.model, client, tokenizer, max_input_tokens, temperature=temp, max_new_tokens=128, timeout=args.timeout)
        if output == '' or output is None:
            return None

        if args.cot:
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, args.model, client, tokenizer, max_input_tokens, temperature=temp, max_new_tokens=128, timeout=args.timeout)
            if output == '' or output is None:
                return None

        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        return item
    except Exception as e:
        print(f"[ERROR] {item_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    output_model = (args.output_model or args.model).split("/")[-1]
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, output_model + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, output_model + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, output_model + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, output_model + ".jsonl")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    dataset = load_from_disk(str(data_dir))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = [item for item in data_all if item["_id"] not in has_data]

    if not data:
        print("All samples already processed.")
        return

    print(f"Pending: {len(data)}, Completed: {len(has_data)}")

    if len(data) == 0:
        print("No pending data to process!")
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY", timeout=args.timeout)
    tokenizer_path = args.tokenizer or os.environ.get("TOKENIZER_PATH") or os.environ.get("MODEL_PATH") or model_map.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    max_input_tokens = args.max_input_tokens or maxlen_map.get(args.model, 120000)
  
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single, item, args, client, tokenizer, max_input_tokens): item["_id"] for item in data}
        with tqdm(total=len(data), desc="Inference") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        with open(out_file, 'a', encoding='utf-8') as fout:
                            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        success_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing result: {e}")
                    pbar.update(1)

    print(f"Successfully wrote {success_count}/{len(data)} results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="0417-preview")
    parser.add_argument("--output_model", type=str, default=None, help="Name used only for the saved result file.")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR.parents[1] / "datasets/data/longbench_v2"))
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path/name. Defaults to TOKENIZER_PATH, MODEL_PATH, or --model.")
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    parser.add_argument("--workers", "-n", type=int, default=32)
    parser.add_argument("--cot", "-cot", action='store_true')
    parser.add_argument("--no_context", "-nc", action='store_true')
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=6000, help="LLM call timeout in seconds")
    args = parser.parse_args()
    main()
