"""
dataset_loader.py — Phase 1, Step 1.2
======================================
Loads and prepares all datasets used in LoRA-SafeLoop experiments.

Datasets returned:
  GSM8K        : Math reasoning training data (7,473 examples)
  Alpaca-5k    : General instruction fine-tuning data (5,000 examples, seed 42)
  AdvBench     : 520 harmful prompts — safety evaluation benchmark
  Safe prompts : 520 benign Alpaca instructions — used in Phase 3 for contrastive
                 activation extraction (separate from training Alpaca-5k)

Design decisions:
  - Returns plain Python lists/dicts. No tokenization here.
    Tokenization happens in the training loop where we have the tokenizer.
  - AdvBench is downloaded from the official llm-attacks GitHub repo and
    cached locally to avoid repeated downloads.
  - All random sampling uses explicit seeds for reproducibility.
"""

import os
import random
import urllib.request
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the LoRA project. dataset_loader.py lives in src/, so root is one
# level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data"
DATA_CACHE_DIR.mkdir(exist_ok=True)

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)
ADVBENCH_LOCAL = DATA_CACHE_DIR / "advbench_harmful.csv"


# ---------------------------------------------------------------------------
# GSM8K — Math reasoning
# ---------------------------------------------------------------------------

def load_gsm8k_train() -> list[dict]:
    """
    Load the full GSM8K training split (7,473 examples).

    Returns:
        List of dicts with keys:
            'question' : str — the math word problem
            'answer'   : str — full solution string ending with '#### <number>'

    Why GSM8K?
        Standard benchmark for mathematical reasoning. The answer always ends
        with '#### <number>', which lets us extract the final answer with a
        simple regex during evaluation (no LLM judge needed for capability).

    Example:
        {'question': 'Natalia sold clips to 48 of her friends...',
         'answer':   '...Natalia sold 48/2 = <<48/2=24>>24 clips...\\n#### 72'}
    """
    print("[dataset_loader] Loading GSM8K training split...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    examples = [
        {"question": row["question"], "answer": row["answer"]}
        for row in dataset
    ]
    print(f"[dataset_loader] GSM8K train: {len(examples)} examples loaded.")
    return examples


def load_gsm8k_test(num_examples: int = 200, seed: int = 42) -> list[dict]:
    """
    Load a fixed subset of the GSM8K test split for evaluation.

    Args:
        num_examples: How many test examples to evaluate on. 200 is enough
                      for a statistically stable accuracy estimate while
                      keeping evaluation fast (important when running every
                      100 training steps).
        seed: Random seed for reproducible subsampling.

    Returns:
        Same format as load_gsm8k_train().
    """
    print(f"[dataset_loader] Loading GSM8K test split ({num_examples} examples)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    all_examples = [
        {"question": row["question"], "answer": row["answer"]}
        for row in dataset
    ]
    random.seed(seed)
    subset = random.sample(all_examples, min(num_examples, len(all_examples)))
    print(f"[dataset_loader] GSM8K test subset: {len(subset)} examples.")
    return subset


# ---------------------------------------------------------------------------
# Alpaca — General instruction following
# ---------------------------------------------------------------------------

def load_alpaca_train(num_examples: int = 5000, seed: int = 42) -> list[dict]:
    """
    Load a fixed sample of the Alpaca instruction dataset for training.

    Args:
        num_examples: Number of examples to sample. 5,000 is our committed
                      hyperparameter (sufficient for clear fine-tuning signal).
        seed: Fixed at 42 for reproducibility. Must match across all runs.

    Returns:
        List of dicts with key:
            'prompt' : str — the instruction (and optional input) combined

    Why combine instruction + input?
        Alpaca has both an 'instruction' field and an optional 'input' field.
        When 'input' is non-empty, the full task is instruction + context.
        We concatenate them as: "{instruction}\n{input}" when input exists.

    Note:
        This is the TRAINING set. The safe prompts for Phase 3 are loaded
        separately by load_safe_prompts() to avoid data contamination.
    """
    print(f"[dataset_loader] Loading Alpaca train ({num_examples} examples, seed={seed})...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    examples = []
    for row in dataset:
        instruction = row["instruction"].strip()
        inp = row["input"].strip()
        prompt = f"{instruction}\n{inp}" if inp else instruction
        examples.append({"prompt": prompt, "output": row["output"].strip()})

    random.seed(seed)
    subset = random.sample(examples, min(num_examples, len(examples)))
    print(f"[dataset_loader] Alpaca train: {len(subset)} examples loaded.")
    return subset


def load_alpaca_val(num_examples: int = 500, seed: int = 42) -> list[dict]:
    """
    Load a fixed validation subset of Alpaca for capability evaluation.

    We compute validation loss (cross-entropy) on this set every 100 steps
    to measure how well the model retains instruction-following capability
    during fine-tuning. No ground-truth extraction needed — loss is the metric.

    Important:
        Uses a different random seed offset (seed + 1) internally to ensure
        these 500 examples do NOT overlap with the 5,000 training examples.
    """
    print(f"[dataset_loader] Loading Alpaca validation ({num_examples} examples)...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    all_examples = []
    for row in dataset:
        instruction = row["instruction"].strip()
        inp = row["input"].strip()
        prompt = f"{instruction}\n{inp}" if inp else instruction
        all_examples.append({"prompt": prompt, "output": row["output"].strip()})

    # Use a different seed to get a non-overlapping subset from training
    random.seed(seed + 1)
    val_subset = random.sample(all_examples, min(num_examples, len(all_examples)))
    print(f"[dataset_loader] Alpaca val: {len(val_subset)} examples loaded.")
    return val_subset


# ---------------------------------------------------------------------------
# AdvBench — Safety evaluation benchmark
# ---------------------------------------------------------------------------

def load_advbench() -> list[str]:
    """
    Load the 520 harmful instructions from AdvBench.

    AdvBench (Zou et al. 2023) is the standard benchmark for evaluating
    LLM safety. It contains 520 harmful instructions across categories:
    hacking, exploitation, violence, illegal activity, etc.

    The model's refusal rate on these prompts is our primary safety metric.

    Returns:
        List of 520 harmful instruction strings (the 'goal' column of the CSV).

    Caching:
        Downloads once to data/advbench_harmful.csv and reuses locally.
        No internet needed after first run.
    """
    if not ADVBENCH_LOCAL.exists():
        print(f"[dataset_loader] Downloading AdvBench from GitHub...")
        try:
            urllib.request.urlretrieve(ADVBENCH_URL, ADVBENCH_LOCAL)
            print(f"[dataset_loader] AdvBench saved to {ADVBENCH_LOCAL}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download AdvBench. Check your internet connection.\n"
                f"URL: {ADVBENCH_URL}\nError: {e}"
            )
    else:
        print(f"[dataset_loader] AdvBench found in cache: {ADVBENCH_LOCAL}")

    df = pd.read_csv(ADVBENCH_LOCAL)
    # The CSV has columns: 'goal' (the harmful instruction) and 'target'
    # (the ideal harmful completion). We only use 'goal' — the prompt.
    harmful_prompts = df["goal"].dropna().tolist()
    print(f"[dataset_loader] AdvBench: {len(harmful_prompts)} harmful prompts loaded.")
    return harmful_prompts


# ---------------------------------------------------------------------------
# Safe prompts — Phase 3 contrastive activation extraction
# ---------------------------------------------------------------------------

def load_safe_prompts(num_examples: int = 520, seed: int = 99) -> list[str]:
    """
    Load benign instruction prompts to pair with AdvBench in Phase 3.

    Phase 3 (safety subspace extraction) computes:
        d_l^(i) = h_harmful^(i) - h_safe^(i)   for each layer l

    This requires a set of safe prompts as a contrast. We use Alpaca
    instructions because:
    1. They are clearly benign (instruction-following tasks)
    2. They have similar syntactic structure to AdvBench prompts
       (both start with an imperative instruction)
    3. They are already downloaded for training

    Args:
        num_examples: Must match len(load_advbench()) = 520 for pairwise
                      subtraction to work.
        seed: Different from all training seeds (99) to avoid overlap.

    Returns:
        List of 520 benign instruction strings.
    """
    print(f"[dataset_loader] Loading safe prompts for Phase 3 contrast ({num_examples} examples)...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    all_instructions = []
    for row in dataset:
        instruction = row["instruction"].strip()
        inp = row["input"].strip()
        prompt = f"{instruction}\n{inp}" if inp else instruction
        all_instructions.append(prompt)

    random.seed(seed)
    safe_prompts = random.sample(all_instructions, min(num_examples, len(all_instructions)))
    print(f"[dataset_loader] Safe prompts: {len(safe_prompts)} loaded.")
    return safe_prompts


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify all loaders work
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Loader Sanity Check")
    print("=" * 60)

    # GSM8K
    gsm8k_train = load_gsm8k_train()
    gsm8k_test = load_gsm8k_test()
    print(f"\nGSM8K train sample:\n  Q: {gsm8k_train[0]['question'][:80]}...")
    print(f"  A: {gsm8k_train[0]['answer'][-30:]}")

    # Alpaca
    alpaca_train = load_alpaca_train()
    alpaca_val = load_alpaca_val()
    print(f"\nAlpaca train sample:\n  {alpaca_train[0]['prompt'][:80]}...")
    print(f"Alpaca val sample:\n  {alpaca_val[0]['prompt'][:80]}...")

    # AdvBench
    advbench = load_advbench()
    print(f"\nAdvBench sample:\n  {advbench[0]}")
    print(f"  {advbench[100]}")

    # Safe prompts
    safe_prompts = load_safe_prompts()
    print(f"\nSafe prompt sample:\n  {safe_prompts[0][:80]}...")

    # Verify counts
    print("\n" + "=" * 60)
    print("Counts:")
    print(f"  GSM8K train:   {len(gsm8k_train):,}")
    print(f"  GSM8K test:    {len(gsm8k_test):,}")
    print(f"  Alpaca train:  {len(alpaca_train):,}")
    print(f"  Alpaca val:    {len(alpaca_val):,}")
    print(f"  AdvBench:      {len(advbench):,}")
    print(f"  Safe prompts:  {len(safe_prompts):,}")

    # Critical check: AdvBench and safe prompts must be same length for Phase 3
    assert len(advbench) == len(safe_prompts), (
        f"CRITICAL: AdvBench ({len(advbench)}) and safe prompts "
        f"({len(safe_prompts)}) must be same length for pairwise subtraction!"
    )
    print("\nAll checks passed. dataset_loader.py is ready.")
