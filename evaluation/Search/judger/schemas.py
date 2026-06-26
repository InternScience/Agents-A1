extracted_answer_format_for_confidence = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
            "additionalProperties": False
        },
        "strict": True
    }
}

extracted_answer_format_for_xbench = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "最终答案": {"type": "string"},
                "解释": {"type": "string"},
                "结论": {"type": "string", "enum": ["正确", "错误"]},
            },
            "required": ["最终答案", "解释", "结论"],
            "additionalProperties": False
        },
        "strict": True
    }
}
