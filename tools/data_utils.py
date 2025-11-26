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
"""Data utilities for loading and preparing datasets for training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from transformers import BatchEncoding
    import torch


@dataclass
class EncodingDataset:
    """A minimal dataset wrapper compatible with ``Trainer``.

    This class provides a simple way to wrap a ``BatchEncoding`` object and make it
    compatible with the Hugging Face ``Trainer``. It implements the ``__len__`` and
    ``__getitem__`` methods, which are required by the ``Trainer``.

    Args:
        encoding (BatchEncoding): The ``BatchEncoding`` object to wrap.
    """

    encoding: "BatchEncoding"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.encoding["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """Return a single sample from the dataset."""
        return {k: v[idx] for k, v in self.encoding.items()}


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """Load a JSON Lines file and return a list of dictionaries.

    Each line in the file is expected to be a JSON object. The function will
    parse each line and return a list of the parsed objects.

    Args:
        path (str): The path to the JSON Lines file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary
            represents a sample from the file.
    """
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input")
            code = obj.get("code") or obj.get("output") or obj.get("answer")
            if prompt is None or code is None:
                raise ValueError(f"Data row is missing a prompt or code field: {line}")
            sample: Dict[str, str] = {"prompt": prompt, "code": code}
            for k, v in obj.items():
                if k not in sample:
                    sample[k] = v
            samples.append(sample)
    return samples


def build_training_text(sample: Dict[str, str]) -> str:
    """Build a training text string from a sample dictionary.

    The function creates a formatted string that includes a system prompt, the
    user's prompt, and the assistant's response. The format is designed to be
    used for training a language model.

    Args:
        sample (Dict[str, str]): A dictionary containing the ``prompt`` and ``code``
            for a single sample.

    Returns:
        str: A formatted string that can be used for training.
    """
    system_prompt = "You are an assistant developed by the Qwen team, specializing in writing quantum code."
    assistant_reply = sample["code"]
    analysis = sample.get("analysis")
    if analysis:
        assistant_reply = f"{analysis}\n\n{assistant_reply}"
    return (
        f"<|system|>{system_prompt}<|end|>"
        f"<|user|>{sample['prompt']}<|end|>"
        f"<|assistant|>{assistant_reply}<|end|>"
    )