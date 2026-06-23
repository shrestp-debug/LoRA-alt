# 🚀 Final Production Runs (To Do in Month 4/5)

Before generating the final plots for the research paper, the following tasks must be executed to ensure statistical rigor and to utilize the updated metrics.

### 1. Re-run Phase 2 Baselines (Vanilla LoRA)
- **Why:** The current CSVs were generated with the legacy "Keyword Matching" metric and only used one seed.
- **Action:** Run `experiments/train_vanilla.py` on Kaggle for both tasks, looping over 3 random seeds. This will automatically use your new **Hugging Face Refusal Judge** to evaluate safety accurately.
- **Commands to run later:**
  ```bash
  # Run GSM8K Baseline
  python experiments/train_vanilla.py --task gsm8k --seed 42
  python experiments/train_vanilla.py --task gsm8k --seed 123
  python experiments/train_vanilla.py --task gsm8k --seed 456

  # Run Alpaca Baseline
  python experiments/train_vanilla.py --task alpaca --seed 42
  python experiments/train_vanilla.py --task alpaca --seed 123
  python experiments/train_vanilla.py --task alpaca --seed 456
  ```

### 2. Run Phase 4 & 5 (SafeLoRA & SafeLoop Algorithm) with 3 Seeds
- **Why:** To prove that your custom SafeLoop algorithm is consistently better than the baseline across different data shuffles.
- **Action:** Once the SafeLoop algorithm is built, run its training script across the same three seeds (`42, 123, 456`). It will also use the **Hugging Face Refusal Judge** to score itself.

### 3. Generate Final Paper Plots
- **Why:** The final plots in `results/drift_curves.png` need to show the Mean ± Standard Deviation shaded areas.
- **Action:** Update the plotting script to aggregate the 3 CSVs per task, and plot the variance bands.

---
*Note: Until we reach this stage, we will continue developing Phase 3, 4, and 5 locally using just Seed 42 to save time and compute.*
