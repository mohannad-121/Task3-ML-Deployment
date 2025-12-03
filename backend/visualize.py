from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np


def create_prediction_plot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    output_path: Path,
) -> Path:
    """
    Create a scatter plot of actual vs predicted values and save as PNG.
    Returns the output path.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Actual vs Predicted House Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
