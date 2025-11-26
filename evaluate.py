# Copyright 2024 The Qwen2.5-Coder-7B-Instruct Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluation script for the quantum-specialized model.

This script can infer from a fine-tuned model on a set of evaluation questions,
generate the corresponding quantum code, and attempt to compile and run it to
verify its validity.

Evaluation process:

1. Read a JSONL file containing several evaluation samples. Each record should
   contain a "prompt" field (the problem description) and an optional "reference"
   field (the reference code).
2. Load the specified model and tokenizer.
3. Generate code for each "prompt" using the model.
4. Attempt to compile the generated code using Python's built-in `compile()`
   function. If compilation is successful, the sample is considered valid.
5. If the `--functional-correctness` flag is set, the script will also execute
   the generated code and compare its output to the reference solution.
6. Calculate the proportion of successfully compiled and functionally correct
   samples and optionally output the generated code to a specified file for
   manual inspection.

Usage example:

```
python evaluate.py \
    --model_dir outputs/peft_qwen25_coder_quantum \
    --eval_file data/eval.jsonl \
    --max_new_tokens 256 \
    --result_file outputs/eval_results.jsonl \
    --functional-correctness
```

Note: Before running this script, ensure that you have installed the necessary
dependencies such as transformers and qiskit, and have sufficient VRAM or RAM
for inference.
"""

import argparse
import json
import logging
from typing import Dict, List

from tools.data_utils import load_jsonl

logger = logging.getLogger(__name__)


def compile_python_code(code: str) -> bool:
    """Attempt to compile the given Python code string.

    Returns True if compilation is successful, otherwise False for syntax errors.
    """
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def execute_python_code(code: str, reference_solution: str) -> bool:
    """Execute the given Python code string and compare its output to the reference solution.

    Returns True if the output matches the reference solution, otherwise False.
    """
    try:
        # We assume that the reference solution is a string that can be evaluated.
        expected_output = eval(reference_solution)
        # We assume that the generated code prints its output to stdout.
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        exec(code)
        sys.stdout = old_stdout
        actual_output = eval(captured_output.getvalue())
        return actual_output == expected_output
    except Exception:
        return False


def main():
    """The main function for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on quantum tasks")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The directory of the fine-tuned model.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="The evaluation dataset in JSONL format.",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=False,
        help="The path to save the generation results.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--functional-correctness",
        action="store_true",
        help="Enable functional correctness evaluation.",
    )
    args = parser.parse_args()

    # Defer importing transformers.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = load_jsonl(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)

    valid_count = 0
    correct_count = 0
    results: List[Dict] = []
    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        system_prompt = "You are an assistant developed by the Qwen team, specializing in writing quantum code."
        input_text = (
            f"<|system|>{system_prompt}<|end|>"
            f"<|user|>{prompt}<|end|>"
            f"<|assistant|>"
        )
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        # Decode the generated tokens, removing the prompt part.
        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
        compile_success = compile_python_code(generated_code)
        if compile_success:
            valid_count += 1

        functional_success = False
        if compile_success and args.functional_correctness and "reference" in sample:
            functional_success = execute_python_code(generated_code, sample["reference"])
            if functional_success:
                correct_count += 1

        results.append({
            "prompt": prompt,
            "generated_code": generated_code,
            "compile_success": compile_success,
            "functional_success": functional_success,
        })
        print(
            f"[{idx+1}/{len(samples)}] Compile {'OK' if compile_success else 'FAIL'}, "
            f"Functional {'OK' if functional_success else 'FAIL'}"
        )

    valid_ratio = valid_count / len(samples)
    print(f"\nValid code ratio: {valid_ratio:.2%}")
    if args.functional_correctness:
        correct_ratio = correct_count / len(samples)
        print(f"Functional correctness ratio: {correct_ratio:.2%}")

    if args.result_file:
        with open(args.result_file, "w", encoding="utf-8") as f_out:
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
