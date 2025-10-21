"""
该脚本实现了对预训练模型的监督微调（Supervised Fine‑Tuning，简称 SFT）。

它主要依赖 Hugging Face Transformers 框架，并假定用户已经安装了 transformers、datasets 等库，并提前下载了 Qwen2.5‑Coder‑7B‑Instruct 模型权重。训练将在 NPU 或 GPU 上进行，具体设备由 transformers 自动检测。

使用示例：

```
python train_sft.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/train.jsonl \
    --validation_file data/valid.jsonl \
    --output_dir outputs/sft_qwen25_coder_quantum \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1
```

训练数据应为 JSON Lines 格式，每行包含至少一个名为 `prompt` 的字段和一个名为 `code` 或 `completion` 的字段。`prompt` 字段用于描述任务，`code` 字段为期望的代码输出。

注意：
  - 请根据实际硬件资源调整 batch_size、梯度累积步数等参数。
  - 为了在没有 transformers 安装的环境下保证脚本可以被正确导入，此脚本将 transformers 的导入放在 `main` 函数内部。
  - 如果在某些环境中没有安装 torchnpu 库，请自行将训练脚本迁移到合适的设备上运行。
"""

import argparse
import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """读取 JSON Lines 格式的数据文件。

    Args:
        path: 数据文件路径。
    Returns:
        列表，每个元素是包含 prompt 和 code 字段的字典。
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # 统一键名为 instruction 和 output，使得数据更通用
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input")
            code = obj.get("code") or obj.get("output") or obj.get("answer")
            if prompt is None or code is None:
                raise ValueError(f"数据行缺少 prompt 或 code 字段: {line}")
            samples.append({"prompt": prompt, "code": code})
    return samples


def prepare_dataset(samples: List[Dict[str, str]], tokenizer):
    """将原始样本转换为用于语言模型训练的格式。

    每条样本将被合成为一个字符串，用于监督训练，例如：

    <|system|>你是通义千问团队开发的助手，擅长编写量子代码。<|end|>
    <|user|>{prompt}<|end|>
    <|assistant|>{code}<|end|>

    Args:
        samples: 原始样本列表。
        tokenizer: 用于编码的 tokenizer。
    Returns:
        一个字典，包含 tokenized 的训练样本。
    """
    def build_text(example):
        system_prompt = "你是通义千问团队开发的助手，擅长编写量子代码。"
        return (
            f"<|system|>{system_prompt}<|end|>"
            f"<|user|>{example['prompt']}<|end|>"
            f"<|assistant|>{example['code']}<|end|>"
        )

    texts = [build_text(sample) for sample in samples]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Supervised fine‑tuning for Qwen2.5‑Coder on quantum tasks")
    parser.add_argument("--model_name", type=str, required=True, help="预训练模型名称或路径，例如 Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件（JSONL 格式）路径")
    parser.add_argument("--validation_file", type=str, required=False, help="验证集文件（JSONL 格式）路径")
    parser.add_argument("--output_dir", type=str, default="outputs/sft", help="模型保存路径")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的 batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="验证时的 batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练 epoch 数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    args = parser.parse_args()

    # 这里延迟导入 transformers 以避免在没有安装的环境下报错
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    # 加载数据
    train_samples = load_jsonl(args.train_file)
    eval_samples = load_jsonl(args.validation_file) if args.validation_file else None

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    # 准备数据集
    tokenized_train = prepare_dataset(train_samples, tokenizer)
    tokenized_eval = prepare_dataset(eval_samples, tokenizer) if eval_samples else None

    # 数据整理器，按语言模型方式拼接
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps" if tokenized_eval is not None else "no",
        save_steps=500,
        logging_steps=100,
        remove_unused_columns=False,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    # 保存模型和分词器
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()