import os
import sys
from datetime import datetime

from absl import app
from absl import flags
from absl import logging

import evaluation_lib


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


class TeeLogger:
  """
  同时输出到终端和 txt 文件
  """

  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = open(filename, "w", encoding="utf-8")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.terminal.flush()
    self.log.flush()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  input_basename = os.path.basename(_INPUT_RESPONSE_DATA.value)

  if "-responses.jsonl" in input_basename:
      model_name = input_basename.replace("-responses.jsonl", "")
  else:
      model_name = os.path.splitext(input_basename)[0]

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  log_file = os.path.join(
      _OUTPUT_DIR.value,
      f"{model_name}_eval_log_{timestamp}.txt"
  )

  sys.stdout = TeeLogger(log_file)

  print(f"Log file saved to: {log_file}")

  inputs = evaluation_lib.read_prompt_list(_INPUT_DATA.value)
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      _INPUT_RESPONSE_DATA.value)

  for func, output_file_name in [
      (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
      (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
  ]:
    print("Generating")
    logging.info("Generating %s...", output_file_name)

    outputs = []

    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))

    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)

    print(f"Accuracy:{accuracy}")
    logging.info("Accuracy: %f", accuracy)

    output_filename = f"{model_name}-{output_file_name}.jsonl"

    output_path = os.path.join(
        _OUTPUT_DIR.value,
        output_filename
    )

    evaluation_lib.write_outputs(output_path, outputs)

    logging.info("Generated: %s", output_path)

    print("=" * 64)
    print(f"{output_path} Accuracy Scores:")

    evaluation_lib.print_report(outputs)


if __name__ == "__main__":
  app.run(main)