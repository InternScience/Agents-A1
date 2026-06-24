# Must match inference/run_inference.py::APPENDED_TEXT byte-for-byte.
# Inference appends this to each question; evaluation strips it before judging.
APPENDED_TEXT = "\n\nPut your reasoning content inside <think></think>. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags."
