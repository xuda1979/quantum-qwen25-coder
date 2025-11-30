"""
This script implements Reinforcement Learning (RL) fine-tuning
for the Qwen2.5-Coder model.
"""

import argparse
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from peft import LoraConfig, TaskType
from rewards.code_reward import CodeReward
from rewards.code_correctness_reward import CodeCorrectnessReward
from rewards.math_correctness_reward import MathCorrectnessReward


def main():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning fine-tuning")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained model name or path")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training data file (JSONL)")
    parser.add_argument("--output_dir", type=str, default="outputs/rl",
                        help="Output directory")
    parser.add_argument("--reward_function", type=str, default="code_simple",
                        help="Reward function")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1,
                        help="Mini batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Steps")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_peft", action="store_true",
                        help="Use PEFT/LoRA")
    parser.add_argument("--lora_r", type=int, default=16)

    args = parser.parse_args()

    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
    )

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.rename_column("prompt", "query")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"],
                                               return_tensors="pt")[0]
        return sample

    dataset = dataset.map(tokenize, batched=False)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    peft_config = None
    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )

    torch_dtype = torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        peft_config=peft_config,
        torch_dtype=torch_dtype
    )

    # Ref model usually disabled if PEFT is used (adapters disabled)
    ref_model = None
    if not args.use_peft:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name)

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator
    )

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
        batch["response"] = tokenizer.batch_decode(response_tensors,
                                                   skip_special_tokens=True)

        # Compute reward
        if args.reward_function == "code":
            rewards = []
            test_files = batch.get("test_file",
                                   [None] * len(batch["response"]))

            for i, response in enumerate(batch["response"]):
                test_file = test_files[i]
                if not test_file:
                    rewards.append(torch.tensor(0.0))
                else:
                    if not os.path.exists(test_file):
                        rewards.append(torch.tensor(0.0))
                        continue

                    reward_function = CodeCorrectnessReward(test_file)
                    reward = reward_function.get_reward([response])[0]
                    rewards.append(torch.tensor(reward))

        elif args.reward_function == "math":
            reward_function = MathCorrectnessReward()
            rewards = [torch.tensor(output) for output in
                       reward_function.get_reward(batch["response"])]
        elif args.reward_function == "code_simple":
            reward_function = CodeReward()
            rewards = [torch.tensor(output) for output in
                       reward_function.get_reward(batch["response"])]
        else:
            raise ValueError(
                f"Unknown reward function: {args.reward_function}")

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
