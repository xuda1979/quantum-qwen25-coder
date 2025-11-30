"""
This script implements Iterative DPO fine-tuning for the Qwen2.5-Coder model
with dynamic preference dataset generation.
"""

import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from rewards.code_correctness_reward import CodeCorrectnessReward
from rewards.math_correctness_reward import MathCorrectnessReward


def main():
    parser = argparse.ArgumentParser(description="Iterative DPO fine-tuning")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained model name or path")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Prompts file (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="outputs/dpo",
                        help="Output directory")
    parser.add_argument("--reward_function", type=str, default="code",
                        help="Reward function (code, math)")
    parser.add_argument("--num_responses", type=int, default=4,
                        help="Responses per prompt for sampling")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max length for generation")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Epochs per iteration")
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="Number of DPO iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_peft", action="store_true",
                        help="Use PEFT/LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    args = parser.parse_args()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    # Use bfloat16 if available
    torch_dtype = torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=torch_dtype)

    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)

    # Load Prompts
    prompts_dataset = load_dataset("json", data_files=args.prompts_file,
                                   split="train")

    for iteration in range(args.num_iterations):
        print(f"--- Iteration {iteration + 1}/{args.num_iterations} ---")

        # 1. Generate Responses
        print("Generating responses...")
        model.eval()
        preference_data = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        for prompt_data in prompts_dataset:
            if args.reward_function == "code":
                if "test_file" not in prompt_data:
                    continue
                # Assuming test_file path is correct relative to cwd
                reward_fn = CodeCorrectnessReward(prompt_data["test_file"])
            elif args.reward_function == "math":
                reward_fn = MathCorrectnessReward()
            else:
                continue

            prompt_text = prompt_data["prompt"]
            input_ids = tokenizer.encode(prompt_text,
                                         return_tensors="pt").to(device)

            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_length,
                    num_return_sequences=args.num_responses,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            responses = []
            for out in outputs:
                generated_text = tokenizer.decode(out[len(input_ids[0]):],
                                                  skip_special_tokens=True)
                responses.append(generated_text)

            # Score
            rewards = reward_fn.get_reward(responses)

            # Pair best and worst
            paired = sorted(zip(rewards, responses), key=lambda x: x[0],
                            reverse=True)
            best_reward, best_response = paired[0]
            worst_reward, worst_response = paired[-1]

            # Filter identical responses or trivial cases if desired
            if best_response == worst_response:
                continue

            # Filter if equal scores
            if best_reward == worst_reward:
                continue

            preference_data.append({
                "prompt": prompt_text,
                "chosen": best_response,
                "rejected": worst_response,
            })

        if not preference_data:
            print("No preference data generated. Skipping training.")
            continue

        print(f"Generated {len(preference_data)} preference pairs.")

        # 2. Train
        print("Training...")
        train_dataset = Dataset.from_list(preference_data)

        training_args = DPOConfig(
            output_dir=f"{args.output_dir}/iter_{iteration}",
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            remove_unused_columns=False,
            bf16=(torch_dtype == torch.bfloat16),
            logging_steps=10,
        )

        # Re-initialize DPOTrainer
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,  # Handles PEFT automatically
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        dpo_trainer.train()
        dpo_trainer.save_model(f"{args.output_dir}/iter_{iteration}")

    print("Iterative DPO completed.")


if __name__ == "__main__":
    main()
