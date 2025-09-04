# visualize.py
import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

def plot_heatmap(csv_path: str, out_path: Optional[str] = None):
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        print("⚠️ Empty CSV; skipping heatmap.")
        return

    # --- Dynamic but safe sizing ---
    n_rows, n_cols = df.shape
    max_width, max_height = 40, 25   # hard caps in inches
    width = min(max_width, max(8, 0.3 * n_cols))
    height = min(max_height, max(6, 0.3 * n_rows))

    plt.figure(figsize=(width, height))
    im = plt.imshow(df.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Frequency")

    # Tick labels
    plt.yticks(range(len(df.index)), df.index, fontsize=6)
    plt.xticks(range(len(df.columns)), df.columns, rotation=90, fontsize=6)

    plt.title("Topic Trend Heatmap", fontsize=12)
    plt.tight_layout()

    if not out_path:
        out_path = csv_path.replace(".csv", ".png")
    plt.savefig(out_path, dpi=150)   # moderate DPI to keep file size reasonable
    plt.close()
    print(f"🖼️ Saved heatmap: {out_path}")
