"""
This script implements Reinforcement Learning (RL) fine-tuning for the Qwen2.5-Coder model,
specifically for quantum programming tasks. It uses the Proximal Policy Optimization (PPO)
algorithm from the TRL library to optimize the model's policy based on a reward signal.

Usage:

python train_rl.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/processed/train.jsonl \
    --output_dir outputs/rl_qwen25_coder_quantum
"""

import argparse
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from rewards.code_reward import CodeReward
from rewards.code_correctness_reward import CodeCorrectnessReward
from rewards.math_correctness_reward import MathCorrectnessReward

def main():
    parser = argparse.ArgumentParser(description="Reinforcement Learning fine-tuning for Qwen2.5-Coder")
    parser.add_argument("--model_name", type=str, required=True, help="Pre-trained model name or path")
    parser.add_argument("--train_file", type=str, required=True, help="Training data file (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="outputs/rl", help="Output directory for the trained model")
    parser.add_argument("--reward_function", type=str, default="code_simple", help="Reward function to use (code, math, etc.)")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Mini batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # PPO configuration
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
    )

    # Load and prepare the dataset
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.rename_column("prompt", "query")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"], return_tensors="pt")[0]
        return sample

    dataset = dataset.map(tokenize, batched=False)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset=dataset, data_collator=collator)

    output_min_length = 4
    output_max_length = 128
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute reward
        if args.reward_function == "code":
            rewards = []
            for i, response in enumerate(batch["response"]):
                if "test_file" not in batch or not batch["test_file"][i]:
                    rewards.append(torch.tensor(0.0))
                else:
                    reward_function = CodeCorrectnessReward(batch["test_file"][i])
                    reward = reward_function.get_reward([response])[0]
                    rewards.append(torch.tensor(reward))
        elif args.reward_function == "math":
            reward_function = MathCorrectnessReward()
            rewards = [torch.tensor(output) for output in reward_function.get_reward(batch["response"])]
        elif args.reward_function == "code_simple":
            reward_function = CodeReward()
            rewards = [torch.tensor(output) for output in reward_function.get_reward(batch["response"])]
        else:
            raise ValueError(f"Unknown reward function: {args.reward_function}")

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # Save the trained model
    ppo_trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
