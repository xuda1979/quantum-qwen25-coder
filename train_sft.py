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
This script implements Supervised Fine-Tuning (SFT) for a pre-trained model.

It relies on the Hugging Face Transformers framework and assumes that the user has
installed the necessary libraries and downloaded the Qwen2.5-Coder-7B-Instruct model
weights. The training will be performed on an NPU or GPU, with the device being
automatically detected by Transformers.

Usage example:

```
python train_sft.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/processed/train.jsonl \
    --validation_file data/processed/valid.jsonl \
    --output_dir outputs/sft_qwen25_coder_quantum \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1
```

The training data should be in JSON Lines format, with each line containing at least
a "prompt" field and a "code" or "completion" field. The "prompt" field describes
the task, while the "code" field provides the expected code output.

Note:
  - Adjust parameters such as batch_size and gradient_accumulation_steps based on
    the available hardware resources.
  - To ensure that the script can be imported correctly in environments without
    Transformers installed, the import of Transformers is placed inside the `main`
    function.
  - If the `torchnpu` library is not installed in your environment, please adapt
    the training script to run on a suitable device.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from tools.data_utils import EncodingDataset, build_training_text, load_jsonl

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_TRAIN_FILE = DEFAULT_PROCESSED_DIR / "train.jsonl"
DEFAULT_VALID_FILE = DEFAULT_PROCESSED_DIR / "valid.jsonl"

logger = logging.getLogger(__name__)


def _maybe_relaunch_with_torchrun(requested_npus: int) -> None:
    """Re-launch the current script under ``torchrun`` if multi-NPU training was requested.

    Hugging Face's ``Trainer`` automatically enables distributed training when the
    process is spawned via ``torchrun``/``torch.distributed``.  When the user
    passes ``--npu <N>`` with ``N > 1`` we re-exec the script under ``torchrun`` to
    spawn ``N`` worker processes.  The re-launch is skipped when the script is
    already running inside a distributed context (``WORLD_SIZE`` > 1).
    """

    if requested_npus <= 1:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError(
            "`torchrun` executable not found. Please install PyTorch with distributed "
            "training support to enable multi-NPU training."
        )

    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")
    env.setdefault("MULTI_NPU_LAUNCHED", "1")
    env.setdefault("ACCELERATE_USE_NPU", "1")
    device_list = ",".join(str(i) for i in range(requested_npus))
    env.setdefault("ASCEND_VISIBLE_DEVICES", device_list)
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", device_list)

    script_args: List[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--npu":
            skip_next = True
            continue
        if arg.startswith("--npu="):
            continue
        script_args.append(arg)

    cmd = [torchrun, "--nproc_per_node", str(requested_npus), sys.argv[0], *script_args]
    subprocess.check_call(cmd, env=env)
    sys.exit(0)


def _set_single_npu_environment() -> None:
    """Ensure single-process runs leverage NPU when ``torch.npu`` is available."""

    if not hasattr(torch, "npu"):
        raise RuntimeError(
            "PyTorch NPU support is not available. Install the Ascend PyTorch build to "
            "run with --npu."
        )

    os.environ.setdefault("ACCELERATE_USE_NPU", "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)


def prepare_dataset(samples: List[Dict[str, str]], tokenizer):
    """Convert raw samples into a format suitable for language model training.

    Each sample will be synthesized into a single string for supervised training,
    for example:

    <|system|>You are an assistant developed by the Qwen team, specializing in writing quantum code.<|end|>
    <|user|>{prompt}<|end|>
    <|assistant|>{code}<|end|>

    Args:
        samples: A list of raw samples.
        tokenizer: The tokenizer to use for encoding.

    Returns:
        A dictionary containing the tokenized training samples.
    """
    texts = [build_training_text(sample) for sample in samples]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    return tokenized


def main():
    """The main function for the SFT training script."""
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for Qwen2.5-Coder on quantum tasks")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name or path of the pre-trained model, e.g., Qwen/Qwen2.5-Coder-7B-Instruct.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(DEFAULT_TRAIN_FILE),
        help="The path to the training data file (in JSONL format), defaults to data/processed/train.jsonl.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=str(DEFAULT_VALID_FILE),
        help="The path to the validation set file (in JSONL format), defaults to data/processed/valid.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sft",
        help="The path to save the model.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="The batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="The batch size for evaluation.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="The learning rate.",
    )
    parser.add_argument(
        "--npu",
        type=int,
        default=0,
        metavar="N",
        help="Use N NPUs for training; if N > 1, the script will be automatically relaunched via torchrun.",
    )
    parser.add_argument(
        "--mock-model",
        action="store_true",
        help="Use a mock model for testing purposes.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    use_npu_backend = bool(args.npu or os.environ.get("ACCELERATE_USE_NPU") == "1")

    _maybe_relaunch_with_torchrun(args.npu)

    if use_npu_backend:
        _set_single_npu_environment()

    train_path = Path(args.train_file).expanduser()
    if not train_path.exists():
        raise FileNotFoundError(
            f"The training data file '{train_path}' does not exist. Please first run "
            f"`python tools/prepare_pdf_dataset.py --pdf-dir <PDF directory>` to generate the data, "
            f"or explicitly specify the data path with the --train_file parameter."
        )
    args.train_file = str(train_path)

    if args.validation_file:
        valid_path = Path(args.validation_file).expanduser()
        if valid_path.exists():
            args.validation_file = str(valid_path)
        else:
            logger.warning(
                "The validation set file '%s' does not exist, skipping evaluation. You can generate it "
                "with prepare_pdf_dataset.py or specify it with --validation_file.",
                valid_path,
            )
            args.validation_file = None

    # Load the data.
    train_samples = load_jsonl(args.train_file)
    eval_samples = load_jsonl(args.validation_file) if args.validation_file else None

    # Load the model and tokenizer.
    if args.mock_model:
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        model = MagicMock()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    if use_npu_backend:
        model.to("npu")

    # Prepare the datasets.
    tokenized_train = EncodingDataset(prepare_dataset(train_samples, tokenizer))
    tokenized_eval = (
        EncodingDataset(prepare_dataset(eval_samples, tokenizer)) if eval_samples else None
    )

    # Data collator for language modeling.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_steps=500,
        logging_steps=100,
        remove_unused_columns=False,
        fp16=True,
        ddp_backend="hccl" if use_npu_backend else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    # Save the model and tokenizer.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
