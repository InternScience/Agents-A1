import json
import os
import http.client
from typing import List, Optional, Union
import requests

from qwen_agent.tools.base import BaseTool, register_tool

SERPER_KEY = os.environ.get('SERPER_KEY_ID')


def contains_chinese_basic(text: str) -> bool:
    return any('\u4E00' <= char <= '\u9FFF' for char in text)


@register_tool("google_search", allow_overwrite=True)
class Search(BaseTool):
    name = "google_search"
    description = "\n    Searches Google using the Serper API and returns a formatted string with the results.\n\n    This tool requires a SERPER_API_KEY to be provided.\n    The response is formatted as a Markdown string, summarizing the search results,\n    which is suitable for consumption by large language models.\n    "
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def google_search_with_serp(self, query: str):
        headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json'
        }

        if contains_chinese_basic(query):
            base_payload = {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            }
        else:
            base_payload = {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            }

        all_organic = []
        answer_box_info = ""

        for page_num in range(1, 6):
            payload = json.dumps({**base_payload, "page": page_num})

            results = None
            for attempt in range(5):
                try:
                    response = requests.request("POST", "https://google.serper.dev/search", headers=headers, data=payload)
                    results = response.json()
                    break
                except Exception as e:
                    print(e)
                    if attempt == 4:
                        if page_num == 1:
                            return f"Google search Timeout, return None, Please try again later."
                        results = None
                    continue

            if results is None or "organic" not in results:
                break

            if page_num == 1 and "answerBox" in results:
                answer = results["answerBox"]
                title = answer.get("title", "Answer")
                snippet = answer.get("snippet", "N/A").replace("\n", " ")
                answer_box_info = f"## {title}\n{snippet}\n\n"

            page_organic = results["organic"]
            all_organic.extend(page_organic)

            if len(page_organic) < 10:
                break

        try:
            if not all_organic:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            organic_info = ""
            for i, item in enumerate(all_organic[:50], 1):
                title = item.get("title", "N/A")
                link = item.get("link", "#")
                snippet = item.get("snippet", "N/A").replace("\n", " ")
                organic_info += f"{i}. **[{title}]({link})**\n   - {snippet}\n"

            content = f"# Search Results for: '{query}'\n\n {answer_box_info} ## Web Results\n\n" + organic_info
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = self.google_search_with_serp(query)
        else:
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.google_search_with_serp(q))
            response = "\n=======\n".join(responses)

        return response
