import json
import os
import time
from typing import List, Union

import requests
import tiktoken
import httpx
from openai import OpenAI
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


@register_tool('read_page', allow_overwrite=True)
class Visit(BaseTool):
    name = 'read_page'
    description = "\n    Read webpage content and generate AI-powered summaries based on user goals.\n    \n    This tool fetches webpage content using Jina API and then uses GPT to extract \n    and summarize information that's relevant to the specified user goal.\n    "
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
            }
        },
        "required": ["url", "goal"]
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[read_page] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        if isinstance(url, str) and url.startswith('['):
            url = url[2:-2]

        if isinstance(url, str):
            response = self.readpage_jina(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url:
                if time.time() - start_time > 900:
                    cur_response = self._format_error_response(u, goal)
                else:
                    try:
                        cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)

        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()

    def call_server(self, msgs, max_retries=2):
        api_key = os.environ.get("SUMMARY_API_KEY")
        url_llm = os.environ.get("SUMMARY_API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        client = OpenAI(
            api_key=api_key,
            base_url=url_llm,
            http_client=httpx.Client(verify=False)
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            content = content[left:right+1]
                    return content
            except Exception as e:
                time.sleep(10)
                print(e)
                if attempt == (max_retries - 1):
                    return ""
                continue

    def jina_readpage(self, url: str) -> str:
        jina_api_key = os.getenv("JINA_API_KEYS", "")
        headers = {
            "Authorization": f"Bearer {jina_api_key}",
            "Accept": "text/plain"
        }
        max_retries = 3
        timeout = 500

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(f"Jina API Error: Status {response.status_code}, Response: {response.text}")
                    if attempt == max_retries - 1:
                        return f"[read_page] Failed to read page: HTTP {response.status_code}"
            except Exception as e:
                time.sleep(0.5)
                print(f"Jina API Request Error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return f"[read_page] Failed to read page: {str(e)}"

        return "[read_page] Failed to read page after all retries"

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            if content and not content.startswith("[read_page] Failed") and content != "[read_page] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[read_page] Failed to read page."

    def readpage_jina(self, url: str, goal: str) -> str:
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))
        content = self.html_readpage_jina(url)

        if content and not content.startswith("[read_page] Failed") and content != "[read_page] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            raw = self.call_server(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[read_page] Summary url[{url}] "
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[read_page] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = self.call_server(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 2
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = self.call_server(messages, max_retries=max_retries)
                    parse_retry_times += 1

            if parse_retry_times >= 3:
                useful_information = self._format_error_response(url, goal)
            else:
                useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[read_page] Could not generate valid summary after maximum retries")
                useful_information = "[read_page] Failed to read page"

            return useful_information
        else:
            return self._format_error_response(url, goal)

    def _format_error_response(self, url, goal):
        useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
        useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        return useful_information
