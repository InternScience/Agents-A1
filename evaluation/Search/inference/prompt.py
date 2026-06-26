SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

Current date: """

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Searches Google using the Serper API and returns a formatted string with the results.\n\n    This tool requires a SERPER_API_KEY to be provided.\n    The response is formatted as a Markdown string, summarizing the search results,\n    which is suitable for consumption by large language models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "description": "The search query to ask Google.",
                        "title": "Query",
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read webpage content and generate AI-powered summaries based on user goals.\n    \n    This tool fetches webpage content using Jina API and then uses GPT to extract \n    and summarize information that's relevant to the specified user goal.\n    ",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL string or a list of URLs.",
                        "title": "Url",
                        "type": "string"
                    },
                    "goal": {
                        "description": "The specific goal or objective for reading the webpage(s). This helps the AI focus on extracting relevant information.",
                        "title": "Goal",
                        "type": "string"
                    }
                },
                "required": ["url", "goal"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "PythonInterpreter",
            "description": "Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "description": "The Python code to execute. Remember to use print() statements for any output you want to see.",
                        "type": "string"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

EXTRACTOR_PROMPT = """You are an intelligent webpage content analyzer. Your task is to process webpage content and extract information relevant to the user's specific goal.

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Analysis Requirements**

### 1. Rational Analysis
- Carefully scan the webpage content to identify sections and data points that directly relate to the user's goal
- Determine which parts of the content are most valuable for achieving the stated objective
- Consider both explicit information and implicit context that may be relevant

### 2. Evidence Extraction
- Extract the most relevant and comprehensive information from the content
- Include full original context whenever possible - do not summarize or truncate important details
- Capture specific data, quotes, statistics, links, or other concrete evidence
- Preserve formatting and structure where it adds value
- Extract multiple paragraphs if necessary to provide complete context

### 3. Summary Generation
- Synthesize the extracted information into a clear, logically structured paragraph
- Prioritize information based on its direct contribution to the user's goal
- Maintain accuracy while ensuring readability and coherence
- Highlight key insights or actionable information

## **Output Format**
Respond in JSON format with exactly these three fields:

```json
{{
    "rational": "Explanation of why the identified content sections are relevant to the user's goal",
    "evidence": "Comprehensive extraction of relevant information with full original context",
    "summary": "Organized summary paragraph with logical flow, prioritizing goal-relevant insights"
}}
```

**Important**: Ensure the JSON is valid and properly formatted. Do not include any text outside the JSON structure."""
