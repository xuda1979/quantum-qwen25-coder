"""
This script implements Direct Preference Optimization (DPO) fine-tuning for the Qwen2.5-Coder model
with dynamic preference dataset generation.

Usage:

python train_dpo_dynamic.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --prompts_file data/processed/prompts.jsonl \
    --output_dir outputs/dpo_qwen25_coder \
    --reward_function code
"""

import argparse
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from rewards.code_correctness_reward import CodeCorrectnessReward
from rewards.math_correctness_reward import MathCorrectnessReward

def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization fine-tuning with dynamic dataset generation")
    parser.add_argument("--model_name", type=str, required=True, help="Pre-trained model name or path")
    parser.add_argument("--prompts_file", type=str, required=True, help="Prompts file (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="outputs/dpo", help="Output directory for the trained model")
    parser.add_argument("--reward_function", type=str, default="code", help="Reward function to use (code, math, etc.)")
    parser.add_argument("--num_responses", type=int, default=2, help="Number of responses to generate for each prompt")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum length of the generated responses")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    prompts_dataset = load_dataset("json", data_files=args.prompts_file, split="train")

    preference_data = []
    for prompt in prompts_dataset:
        if args.reward_function == "code":
            if "test_file" not in prompt:
                continue
            reward_function = CodeCorrectnessReward(prompt["test_file"])
        elif args.reward_function == "math":
            reward_function = MathCorrectnessReward()
        else:
            raise ValueError(f"Unknown reward function: {args.reward_function}")

        responses = []
        for _ in range(args.num_responses):
            input_ids = tokenizer.encode(prompt["prompt"], return_tensors="pt")
            output = model.generate(input_ids, max_length=args.max_length, num_return_sequences=1)
            generated_tokens = output[0][len(input_ids[0]):]
            responses.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))

        rewards = reward_function.get_reward(responses)
        sorted_responses = [x for _, x in sorted(zip(rewards, responses), reverse=True)]

        preference_data.append({
            "prompt": prompt["prompt"],
            "chosen": sorted_responses[0],
            "rejected": sorted_responses[-1],
        })

    preference_dataset = Dataset.from_list(preference_data)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        report_to="none",
        bf16=False,
        fp16=False,
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=0.1,
        train_dataset=preference_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()

    dpo_trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
