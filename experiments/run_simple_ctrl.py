"""
run_simple_ctrl.py — Phase 5
======================================
Custom training loop for the Rule-Based Adaptive Controller (B4).
Dynamically scales the SaLoRA-style projection based on observed refusal rate.

Usage:
  python experiments/run_simple_ctrl.py --task alpaca --seed 42
  python experiments/run_simple_ctrl.py --task gsm8k --seed 42
"""

import os
import sys
import argparse
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import contextlib

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    load_gsm8k_train,
    load_gsm8k_test,
    load_alpaca_train,
    load_alpaca_val,
    load_advbench
)
from src.metrics import evaluate_task_gsm8k, evaluate_task_alpaca, evaluate_safety
from src.simple_controller import FeedbackController, ControllerConfig

# We reuse the dataset formatting and collate logic from train_vanilla
from experiments.train_vanilla import MaskedTrainingDataset, training_collate_fn, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Rule-Based Adaptive Controller (B4)")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "alpaca"],
                        help="Task to train on: 'gsm8k' or 'alpaca'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--eval_every", type=int, default=100, help="Step interval for eval")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    # Controller Hyperparameters
    parser.add_argument("--threshold_low", type=float, default=0.85, help="Increase alpha if refusal drops below this")
    parser.add_argument("--threshold_high", type=float, default=0.95, help="Decrease alpha if refusal rises above this")
    parser.add_argument("--delta", type=float, default=0.05, help="Step size for alpha increments")
    
    args = parser.parse_args()

    set_seed(args.seed)

    # Output paths setup
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    csv_path = results_dir / f"simplectrl_{args.task}_seed{args.seed}.csv"
    adapter_save_dir = models_dir / f"simplectrl_{args.task}_seed{args.seed}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running FeedbackController on device: {device}")

    safety_directions_path = models_dir / "safety_directions.pt"
    assert safety_directions_path.exists(), "Safety directions must be extracted (Phase 3) before running Phase 5!"
    safety_directions = torch.load(safety_directions_path, map_location=device, weights_only=True)
    logger.info("Loaded fixed safety directions from models/safety_directions.pt")

    # 1. Load Tokenizer & Datasets
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.task == "gsm8k":
        raw_train_data = load_gsm8k_train()
        eval_task_data = load_gsm8k_test(num_examples=200, seed=args.seed)
    else:
        raw_train_data = load_alpaca_train(num_examples=5000, seed=args.seed)
        eval_task_data = load_alpaca_val(num_examples=500, seed=args.seed)

    advbench_prompts = load_advbench()[:100]

    train_dataset = MaskedTrainingDataset(raw_train_data, tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=lambda b: training_collate_fn(b, tokenizer),
        generator=generator
    )

    # 2. Load Base Model and Apply LoRA
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        model_dtype = torch.float32
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=model_dtype, low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    # --- Phase 5: Initialize the FeedbackController ---
    ctrl_config = ControllerConfig(
        threshold_low=args.threshold_low, 
        threshold_high=args.threshold_high, 
        delta=args.delta
    )
    controller = FeedbackController(model, safety_directions, config=ctrl_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    grad_accumulation_steps = 4
    total_steps = args.steps
    eval_every = args.eval_every
    
    history = []
    step = 0
    epoch = 0
    running_loss = 0.0
    accumulated_loss = 0.0
    batch_idx = 0
    data_iter = iter(train_loader)

    model.zero_grad()

    while step < total_steps:
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            train_loader.generator.manual_seed(args.seed + epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )
        
        loss = outputs.loss / grad_accumulation_steps
        loss.backward()

        accumulated_loss += loss.item() * grad_accumulation_steps
        running_loss += loss.item() * grad_accumulation_steps
        batch_idx += 1

        if batch_idx % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            current_step = step + 1
            if current_step % 10 == 0:
                logger.info(f"Step {current_step}/{total_steps} | Training Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

            if current_step % eval_every == 0:
                logger.info(f"--- Running Evaluation at Step {current_step} ---")
                
                # Evaluate using local HF Refusal Judge
                refusal_rate = evaluate_safety(
                    model, tokenizer, advbench_prompts, batch_size=4, device=device
                )

                if args.task == "gsm8k":
                    task_metric = evaluate_task_gsm8k(model, tokenizer, eval_task_data, batch_size=4, device=device)
                    metric_name = "gsm8k_accuracy"
                else:
                    task_metric = evaluate_task_alpaca(model, tokenizer, eval_task_data, batch_size=4, device=device)
                    metric_name = "alpaca_val_loss"

                # Update alpha based on observed refusal_rate
                # This alpha is held constant until the next evaluation (eval_every steps)
                current_alpha = controller.step(refusal_rate)

                record = {
                    "step": current_step,
                    "train_loss": accumulated_loss / eval_every,
                    "refusal_rate": refusal_rate,
                    metric_name: task_metric,
                    "alpha": current_alpha
                }
                history.append(record)
                
                df = pd.DataFrame(history)
                df.to_csv(csv_path, index=False)
                
                checkpoint_dir = adapter_save_dir / f"checkpoint-{current_step}"
                model.save_pretrained(checkpoint_dir)
                
                accumulated_loss = 0.0

            step += 1

    model.save_pretrained(adapter_save_dir)
    logger.info(f"Training complete. Adapter saved to {adapter_save_dir}")

if __name__ == "__main__":
    main()
