import os
import numpy as np
import matplotlib.pyplot as plt
from data_processing import BOLDDataset

# ---- CONFIG ----
BOLD_ROOT = '/Users/RyoSeah/Desktop/Stress_Detection/BOLD_public'
SEQUENCE_LENGTH = 30
OVERLAP_RATIO = 0.5
MIN_CONFIDENCE = 0.3

def main():
    # Initialize loader
    loader = BOLDDataset(
        bold_root=BOLD_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        overlap_ratio=OVERLAP_RATIO,
        min_confidence=MIN_CONFIDENCE
    )

    # Load all stress scores from train and val splits
    print("Loading train split...")
    _, y_train = loader.load_dataset('train')
    print("Loading val split...")
    _, y_val = loader.load_dataset('val')

    all_scores = np.concatenate([y_train, y_val])
    print(f"Total stress scores: {len(all_scores)}")
    print(f"Min: {all_scores.min():.3f}, Max: {all_scores.max():.3f}, Mean: {all_scores.mean():.3f}, Std: {all_scores.std():.3f}")

    # Baseline MAE: always predict the mean
    baseline_pred = np.full_like(all_scores, fill_value=all_scores.mean())
    baseline_mae = np.mean(np.abs(all_scores - baseline_pred))
    print(f"Baseline MAE (predicting mean): {baseline_mae:.4f}")

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(all_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Stress Scores (Train + Val)')
    plt.xlabel('Stress Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    output_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'stress_score_distribution.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()