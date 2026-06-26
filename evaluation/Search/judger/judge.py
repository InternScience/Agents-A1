import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Literal, Tuple

import litellm
from dotenv import dotenv_values, load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from prompt import (
    JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    JUDGE_PROMPT_GAIA,
    JUDGE_PROMPT_HLE,
    JUDGE_PROMPT_SEAL,
    JUDGE_PROMPT_XBENCH,
)
from schemas import (
    extracted_answer_format_for_confidence,
)

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)
DOTENV_VALUES = dotenv_values(ENV_PATH)


def get_env(name: str) -> str:
    return (os.getenv(name) or DOTENV_VALUES.get(name) or "").strip()


thread_local = threading.local()
JUDGE_API_KEY = get_env("JUDGE_API_KEY")
JUDGE_API_BASE = get_env("JUDGE_API_BASE")
JUDGE_MODEL_NAME = get_env("JUDGE_MODEL_NAME")


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


def get_client():
    if not hasattr(thread_local, 'client'):
        if not JUDGE_API_KEY or not JUDGE_API_BASE:
            raise RuntimeError(
                "Judge API key and base URL are not configured. Set JUDGE_API_KEY and JUDGE_API_BASE in .env."
            )
        thread_local.client = OpenAI(
            api_key=JUDGE_API_KEY,
            base_url=JUDGE_API_BASE,
        )
    return thread_local.client


def is_correct_judgement(judgement):
    return judgement.lower() == "correct" or (judgement and judgement.lower()[0] == "a")


def resolve_judge(dataset: str) -> Tuple[str, str]:
    if dataset == "gaia":
        judge_model = "gpt-4o"
        judge_prompt = JUDGE_PROMPT_GAIA
    elif dataset in ["seal-0"]:
        judge_model = "gpt-4.1"
        judge_prompt = JUDGE_PROMPT_SEAL
    elif dataset in ["xbench-deepsearch"]:
        judge_prompt = JUDGE_PROMPT_XBENCH
        judge_model = "google/gemini-2.0-flash-001"
    elif dataset.startswith("browsecomp_zh"):
        judge_model = "gpt-4o-2024-08-06"
        judge_prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL
    elif dataset.startswith("browsecomp_en"):
        judge_model = "gpt-4o-2024-08-06"
        judge_prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL
    elif dataset == "hle":
        judge_model = "openai/o3-mini"
        judge_prompt = JUDGE_PROMPT_HLE
    else:
        judge_model = "gpt-4o"
        judge_prompt = JUDGE_PROMPT_GAIA

    if JUDGE_MODEL_NAME:
        judge_model = JUDGE_MODEL_NAME

    return judge_model, judge_prompt


def call_llm_judge(item, *, judge_prompt, dataset, judge_model):
    question = item["question"]
    correct_answer = item["answer"]
    response = item["prediction"].strip()
    prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)

    for attempt in range(100):
        try:
            if dataset == "xbench-deepsearch":
                judgement = _judge_xbench(prompt, judge_model)

            elif dataset == "hle":
                judgement = _judge_hle(prompt, judge_model)

            elif "browsecomp" in dataset:
                judgement = _judge_browsecomp(prompt, judge_model)

            else:
                # gaia, seal-0, and any default
                judgement = _complete_text(prompt, judge_model)

            return {
                "question": question,
                "answer": correct_answer,
                "judgement": judgement
            }

        except Exception as e:
            print(traceback.format_exc())

            if attempt == 4:
                print(f"Error judgement for question: {question}: {e}")
                return {
                    "question": question,
                    "answer": correct_answer,
                    "judgement": "Error",
                    "error": str(e)
                }
            time.sleep(3)
            continue


def _complete_text(prompt, judge_model):
    response = litellm.completion(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        num_retries=5,
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_API_BASE,
    )
    return response.choices[0].message["content"]


def _judge_xbench(prompt, judge_model):
    raw_content = _complete_text(prompt, judge_model) or ""
    m = re.search(r"结论\s*[:：]\s*['\"]?\s*(正确|错误)", raw_content)
    return "Correct" if (m and m.group(1) == "正确") else "Incorrect"


def _judge_hle(prompt, judge_model):
    client = get_client()
    judgement = "Error"
    for _ in range(6):
        try:
            response_obj = client.beta.chat.completions.parse(
                model=judge_model,
                max_completion_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
                response_format=ExtractedAnswer,
                timeout=60.0,
            )
            parsed = response_obj.choices[0].message.parsed
            judgement = "Correct" if parsed.correct == "yes" else "Incorrect"
            break
        except Exception as e:
            if "length limit" in str(e):
                break
            time.sleep(1)
    return judgement


def _judge_browsecomp(prompt, judge_model):
    response = litellm.completion(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        num_retries=5,
        response_format=extracted_answer_format_for_confidence,
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_API_BASE,
    )
    raw_content = response.choices[0].message["content"]
    raw_judge = json.loads(raw_content)
    return "Correct" if raw_judge["correct"].lower() == "yes" else raw_content
