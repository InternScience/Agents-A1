import json
import json5
import os
import re
import copy
import time
import random
import datetime
from typing import Dict, List, Optional, Union

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message
from qwen_agent.tools import BaseTool
from prompt import SYSTEM_PROMPT, TOOLS
from multimodal import (
    build_user_message,
    flatten_messages_for_tokenization,
    runtime_multimodal_error,
)

from tool_search import *
from tool_visit import *
from tool_python import *

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

regex_pattern = r'[\s\S]*<\/think>'


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def _extract_reasoning(msg):
    """Return (field_name, value) preserving which reasoning field the provider used.

    Different OpenAI-compatible backends emit the chain-of-thought under different
    keys: vLLM/qwen3-parser, DeepSeek and Moonshot use `reasoning_content`; some
    others use `reasoning`. We keep the original field name so any downstream
    consumer that round-trips messages back to the API sees the same shape the
    provider originally produced.
    """
    if getattr(msg, "reasoning_content", None) is not None:
        return ("reasoning_content", msg.reasoning_content)
    if getattr(msg, "reasoning", None) is not None:
        return ("reasoning", msg.reasoning)
    return ("reasoning", "")


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):
        if os.path.exists(llm["model_path"]):
            self.tokenizer = AutoTokenizer.from_pretrained(llm["model_path"])
        else:
            self.tokenizer = None
            print(f"Warning: Model path '{llm['model_path']}' not found locally. Token counting disabled.")
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model_path"]

        print(f"llm_generate_cfg: {self.llm_generate_cfg}")
        print(f"llm_local_path: {self.llm_local_path}")

    def call_server(self, msgs, planning_port, max_tries=10):
        openai_api_key = os.getenv("AGENT_API_KEY", "EMPTY")
        openai_api_base = os.getenv("AGENT_API_BASE_URL", f"http://127.0.0.1:{planning_port}/v1")

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1
        extra_body = {}
        top_k = self.llm_generate_cfg.get("top_k", -1)
        if top_k != -1:
            extra_body["top_k"] = top_k
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                    tools=TOOLS,
                    extra_body=extra_body or None,
                )
                return chat_response.choices[0].message

            except (APIError, APIConnectionError, APITimeoutError) as e:
                err = runtime_multimodal_error(
                    msgs,
                    model_path=self.llm_local_path,
                    served_model_name=self.model,
                    error_text=str(e),
                    base_url=openai_api_base,
                )
                if err:
                    raise RuntimeError(err) from e
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")

        return "vllm server error!!!"

    def count_tokens(self, messages):
        if self.tokenizer is None:
            return 0
        try:
            flat = flatten_messages_for_tokenization(messages)
            full_prompt = self.tokenizer.apply_chat_template(flat, tokenize=False)
            tokens = self.tokenizer(full_prompt, return_tensors="pt")
            return len(tokens["input_ids"][0])
        except Exception:
            return 0

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model = model
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT + today_date()

        round = 0
        user_message = build_user_message(
            data["item"],
            question,
            model_path=self.llm_local_path,
            served_model_name=self.model,
            base_dir=data.get("base_dir"),
        )
        messages = [{"role": "system", "content": system_prompt}, user_message]
        passin_messages = copy.deepcopy(messages)
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN

        while num_llm_calls_available > 0:
            if time.time() - start_time > 150 * 60:  # 150 minutes timeout
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": 'No answer found after 2h30mins',
                    "termination": 'timeout'
                }

            round += 1
            num_llm_calls_available -= 1
            response = self.call_server(passin_messages, planning_port)
            print(response)
            reasoning_field, reasoning = _extract_reasoning(response)
            content = response.content
            tool_calls = response.tool_calls
            print(f"Round {round}\n{reasoning=}\n{content=}\n{tool_calls=}")

            msg_entry = {
                "role": "assistant",
                reasoning_field: reasoning,
                "content": content,
                "tool_calls": [{"type": tc.type, "id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls] if tool_calls else None,
            }
            messages.append(msg_entry)
            passin_messages.append(copy.deepcopy(msg_entry))

            if tool_calls and len(tool_calls) > 0:
                for tool in tool_calls:
                    try:
                        tool_name = tool.function.name
                        tool_args = json5.loads(tool.function.arguments)
                        tool_response = self.custom_call_tool(tool_name, tool_args)
                    except Exception as e:
                        tool_response = 'Error: ' + str(e)
                    tool_msg = {"role": "tool", "tool_call_id": tool.id, "name": tool_name, "content": tool_response}
                    messages.append(tool_msg)
                    passin_messages.append(copy.deepcopy(tool_msg))
            else:
                if content is not None and '<answer>' in content and '</answer>' in content:
                    termination = 'answer'
                    break

                if content is not None and num_llm_calls_available <= 0 and '<answer>' not in content:
                    messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 128 * 1024
            token_count = self.count_tokens(passin_messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                overflow_msg = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                messages[-1]['content'] = overflow_msg
                passin_messages[-1]['content'] = overflow_msg
                choice = self.call_server(passin_messages, planning_port)
                reasoning_field, reasoning = _extract_reasoning(choice.message)
                content = choice.message.content
                tool_calls = choice.message.tool_calls

                overflow_entry = {
                    "role": "assistant",
                    reasoning_field: reasoning,
                    "content": content.strip(),
                    "tool_calls": [{"type": tc.type, "id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls] if tool_calls else None,
                }
                messages.append(overflow_entry)
                passin_messages.append(copy.deepcopy(overflow_entry))

                extracted_answer = re.sub(regex_pattern, '', content)
                if '<answer>' in content and '</answer>' in content:
                    prediction = extracted_answer.split('<answer>')[-1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = passin_messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }

        extracted_answer = messages[-1]['content'].strip()
        if '<answer>' in extracted_answer and '</answer>' in extracted_answer:
            prediction = extracted_answer.split('<answer>')[-1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        return {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name not in TOOL_MAP:
            return f"Error: Tool {tool_name} not found"
        # PythonInterpreter.call() takes the raw code string, not the args dict.
        if "python" in tool_name.lower():
            return TOOL_MAP[tool_name].call(tool_args["code"])
        tool_args["params"] = tool_args
        return TOOL_MAP[tool_name].call(tool_args, **kwargs)
