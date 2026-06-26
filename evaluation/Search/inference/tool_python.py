import re
from typing import Dict, List, Optional, Union, Any
import json5
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
from sandbox_fusion import run_code, RunCodeRequest, RunStatus
from requests.exceptions import Timeout
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CHINESE_CHAR_RE = re.compile(r'[一-鿿]')


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))


# Array of sandbox fusion endpoints
SANDBOX_FUSION_ENDPOINTS = []

# Fallback to single endpoint if environment variable exists.
# SANDBOX_FUSION_ENDPOINT is a comma-separated list of running SandboxFusion
# server URLs, e.g. "http://127.0.0.1:8080,http://127.0.0.1:8081".
if 'SANDBOX_FUSION_ENDPOINT' in os.environ:
    SANDBOX_FUSION_ENDPOINTS = os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')


@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"
    description = 'Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**'

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Remember to use print() statements for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = 'The input for this tool should be a Markdown code block.'
            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    @property
    def function(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    def call(self, params, files=None, timeout=50, **kwargs) -> str:
        """Run Python `code` in a SandboxFusion sandbox.

        `params` is the raw Python source string (the agent's custom_call_tool
        passes tool_args["code"] directly). Returns concatenated stdout/stderr,
        or a clear error string when no endpoint is configured/reachable.
        """
        try:
            code = params
            if not SANDBOX_FUSION_ENDPOINTS:
                return '[Python Interpreter Error]: SANDBOX_FUSION_ENDPOINT is not configured. Set it to a running SandboxFusion server URL.'

            last_error = None
            for attempt in range(5):
                try:
                    # Randomly sample an endpoint for each attempt
                    endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
                    print(f"Attempt {attempt + 1}/5 using endpoint: {endpoint}")

                    code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout), max_attempts=1, client_timeout=timeout, endpoint=endpoint)
                    print("[Python] Code Result", code_result)
                    result = []
                    if code_result.run_result.stdout:
                        result.append(f"stdout:\n{code_result.run_result.stdout}")
                    if code_result.run_result.stderr:
                        result.append(f"stderr:\n{code_result.run_result.stderr}")
                    if code_result.run_result.execution_time >= timeout - 1:
                        result.append(f"[PythonInterpreter Error] TimeoutError: Execution timed out.")
                    result = '\n'.join(result)
                    print('SUCCESS RUNNING TOOL')
                    return result if result.strip() else 'Finished execution.'

                except Timeout as e:
                    last_error = f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}.'
                    print(f"Timeout on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:  # Last attempt
                        return last_error
                    continue

                except Exception as e:
                    last_error = f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint}'
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:  # Last attempt
                        return last_error
                    continue

            return last_error if last_error else '[Python Interpreter Error]: All attempts failed.'

        except Exception as e:
            return f"[Python Interpreter Error]: {str(e)}"

    def call_specific_endpoint(self, params: Union[str, dict], endpoint: str, timeout: Optional[int] = 30, **kwargs) -> tuple:
        """Test a specific endpoint directly"""
        try:
            if type(params) is str:
                params = json5.loads(params)
            code = params.get('code', '')
            if not code:
                code = params.get('raw', '')
            triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if triple_match:
                code = triple_match.group(1)
        except Exception:
            code = extract_code(params)

        if not code.strip():
            return False, '[Python Interpreter Error]: Empty code.'

        try:
            start_time = time.time()
            code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout),
                                 max_attempts=1, client_timeout=timeout, endpoint=endpoint)
            end_time = time.time()

            result = []
            if code_result.run_result.stdout:
                result.append(f"stdout:\n{code_result.run_result.stdout}")
            if code_result.run_result.stderr:
                result.append(f"stderr:\n{code_result.run_result.stderr}")

            result = '\n'.join(result)
            execution_time = end_time - start_time
            return True, result if result.strip() else 'Finished execution.', execution_time

        except Timeout as e:
            return False, f'[Python Interpreter Error] TimeoutError: Execution timed out.', None
        except Exception as e:
            return False, f'[Python Interpreter Error]: {str(e)}', None


if __name__ == "__main__":
    tool = PythonInterpreter()
    params = 'print(123)'
    result = tool.call(params)
    print(result)
