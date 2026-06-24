"""
plot_drift_curves.py — Phase 2 & 4
=========================================
Generates Figure 1 for the paper: the safety drift curves showing
Refusal Rate and Task Capability metrics over training steps for all baselines.

Looks for:
  results/<method>_<task>_seed*.csv

Averages logs over seeds if multiple are found, and plots mean +/- standard deviation.
Saves the resulting plot as results/drift_curves.png.
"""

import re
import logging
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_aggregate(results_dir: Path, method: str, task: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads all logs for a specific method and task, computing mean and std at each step."""
    pattern = re.compile(rf"{method}_{task}_seed(\d+)\.csv")
    files = [f for f in results_dir.glob("*.csv") if pattern.match(f.name)]

    if not files:
        logger.warning(f"No log files found for method: {method}, task: {task}")
        return None, None

    logger.info(f"Found {len(files)} run files for method: {method}, task: {task}")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "step" not in df.columns:
            logger.warning(f"File {f.name} missing 'step' column. Skipping.")
            continue
        dfs.append(df.set_index("step"))

    if not dfs:
        return None, None

    combined = pd.concat(dfs, axis=0, keys=range(len(dfs)))
    mean_df = combined.groupby(level=1).mean()
    std_df = combined.groupby(level=1).std().fillna(0.0)

    return mean_df.reset_index(), std_df.reset_index()


def generate_plots():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    methods = [
        {"id": "vanilla", "label": "Vanilla LoRA", "marker": "o"},
        {"id": "safelora", "label": "SafeLoRA (adapted)", "marker": "s"},
        {"id": "salora", "label": "SaLoRA", "marker": "^"},
        {"id": "simplectrl", "label": "Adaptive Control (Phase 5)", "marker": "D"}
    ]
    
    # Harmonious colors for different methods
    colors_safety = {"vanilla": "#EF4444", "safelora": "#F59E0B", "salora": "#10B981", "simplectrl": "#EC4899"} # Red, Amber, Emerald, Pink
    colors_task = {"vanilla": "#3B82F6", "safelora": "#8B5CF6", "salora": "#06B6D4", "simplectrl": "#F43F5E"}   # Blue, Purple, Cyan, Rose
    
    has_data = False

    def plot_task(ax, task_name, title, metric_col, metric_label):
        nonlocal has_data
        ax_twin = ax.twinx()
        
        ax.set_xlabel("Training Steps", fontsize=11, fontweight='bold')
        ax.set_ylabel("Refusal Rate (Safety)", fontsize=11, fontweight='bold')
        ax_twin.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        
        lines = []
        for m in methods:
            mean_df, std_df = load_and_aggregate(results_dir, m["id"], task_name)
            if mean_df is None:
                continue
                
            has_data = True
            steps = mean_df["step"]
            
            # Plot Safety
            color_s = colors_safety[m["id"]]
            l1 = ax.plot(steps, mean_df["refusal_rate"], color=color_s, linestyle="-", 
                         marker=m["marker"], label=f"{m['label']} (Safety)")
            ax.fill_between(
                steps, 
                np.clip(mean_df["refusal_rate"] - std_df["refusal_rate"], 0.0, 1.0),
                np.clip(mean_df["refusal_rate"] + std_df["refusal_rate"], 0.0, 1.0),
                color=color_s, alpha=0.1
            )
            
            # Plot Task Capability
            color_t = colors_task[m["id"]]
            l2 = ax_twin.plot(steps, mean_df[metric_col], color=color_t, linestyle="--", 
                              marker=m["marker"], label=f"{m['label']} (Capability)")
            ax_twin.fill_between(
                steps, 
                np.clip(mean_df[metric_col] - std_df[metric_col], 0.0, 1.0) if "accuracy" in metric_col else mean_df[metric_col] - std_df[metric_col],
                np.clip(mean_df[metric_col] + std_df[metric_col], 0.0, 1.0) if "accuracy" in metric_col else mean_df[metric_col] + std_df[metric_col],
                color=color_t, alpha=0.1
            )
            
            lines.extend(l1 + l2)

        ax.set_ylim(-0.05, 1.05)
        if "accuracy" in metric_col:
            ax_twin.set_ylim(-0.05, 1.05)
            
        if lines:
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=True, facecolor="white")
            
        ax.set_title(title, fontsize=13, pad=12, fontweight='bold')
        return len(lines) > 0

    # Plot GSM8K
    has_gsm = plot_task(ax1, "gsm8k", "GSM8K (Math reasoning)", "gsm8k_accuracy", "Accuracy (Capability)")
    if not has_gsm:
        ax1.text(0.5, 0.5, "No GSM8K data found.", ha='center', va='center', color='gray')

    # Plot Alpaca
    has_alp = plot_task(ax2, "alpaca", "Alpaca-5k (General instruction)", "alpaca_val_loss", "Validation Loss (Utility)")
    if not has_alp:
        ax2.text(0.5, 0.5, "No Alpaca data found.", ha='center', va='center', color='gray')

    if has_data:
        # Adjust layout to accommodate the external legend
        plt.subplots_adjust(right=0.75, wspace=0.6)
        plot_path = results_dir / "drift_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Successfully saved drift curves plot to {plot_path}")
    else:
        logger.warning("No plot generated: no task datasets were available to aggregate.")
        
    plt.close()

if __name__ == "__main__":
    generate_plots()
