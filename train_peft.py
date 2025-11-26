# Copyright 2024 The Qwen2.5-Coder-7B-Instruct Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you aapt may not use this file except in compliance with the License.
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
This script uses Parameter-Efficient Fine-Tuning (PEFT) to specialize the Qwen2.5-Coder
for quantum programming tasks.

It employs the LoRA (Low-Rank Adaptation) method, which adds a small number of trainable
parameters to specific projection layers while keeping the original model weights frozen.
This significantly reduces memory usage and training costs.

Before running this script, ensure that you have installed the `peft`, `transformers`,
and `datasets` dependencies, and have prepared the training data in JSONL format.
Similar to the other scripts, module imports are done inside functions to accommodate
environments without these dependencies.

Usage example:

```
python train_peft.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/processed/train.jsonl \
    --validation_file data/processed/valid.jsonl \
    --output_dir outputs/peft_qwen25_coder_quantum \
    --target_modules q_proj v_proj \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```
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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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
    """Re-launch the current script under ``torchrun`` when multi-NPU training is requested."""

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
    """Ensure single-process runs leverage ``torch.npu`` when available."""

    if not hasattr(torch, "npu"):
        raise RuntimeError(
            "PyTorch NPU support is not available. Install the Ascend PyTorch build to "
            "run with --npu."
        )

    os.environ.setdefault("ACCELERATE_USE_NPU", "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)


def main():
    """The main function for the PEFT training script."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5-Coder on quantum tasks")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name or path of the pre-trained model.",
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
        help="The path to the validation data file (in JSONL format), defaults to data/processed/valid.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/peft",
        help="The output directory.",
    )
    parser.add_argument(
        "--target_modules",
        nargs="*",
        default=["q_proj", "v_proj"],
        help="The list of module names to apply LoRA to.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="The LoRA rank parameter.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="The LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The LoRA dropout rate.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="The training batch size per device.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="The number of training epochs.",
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

    # Load the datasets.
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

    # Configure PEFT.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Build and encode the training text.
    def preprocess(samples: List[Dict[str, str]]):
        texts = [build_training_text(s) for s in samples]
        encoded = tokenizer(texts, truncation=True, padding=False, return_tensors="pt")
        return EncodingDataset(encoded)

    tokenized_train = preprocess(train_samples)
    tokenized_eval = preprocess(eval_samples) if eval_samples else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-4,
        fp16=True,
        remove_unused_columns=False,
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
