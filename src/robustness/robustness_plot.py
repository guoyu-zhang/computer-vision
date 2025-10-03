import os
import pandas as pd
import matplotlib.pyplot as plt

# Output directory for saving plots
SAVE_DIR = "/Users/bin/Desktop/CV_Assignment/Robustness_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load mean Dice score data
df = pd.read_csv("/Users/bin/Desktop/CV_Assignment/Robustness_Results/mean_dice_scores.csv")

# Define the order of the 8 perturbation types
perturb_order = [
    "gaussian_noise", "gaussian_blur", "contrast_increase", "contrast_decrease",
    "brightness_increase", "brightness_decrease", "occlusion", "salt_pepper"
]

# Define x-axis tick labels for each perturbation type
xtick_labels_dict = {
    "gaussian_noise": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    "gaussian_blur": list(range(10)),
    "contrast_increase": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
    "contrast_decrease": [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
    "brightness_increase": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    "brightness_decrease": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    "occlusion": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    "salt_pepper": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
}

# Font sizes for the plot
title_font = 22
label_font = 20
tick_font = 18

# Create a 2x4 subplot grid
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

# Plot each perturbation type
for i, perturb in enumerate(perturb_order):
    subset = df[df["Perturbation"] == perturb]
    dice_scores = subset["Mean Dice Score"].values
    xtick_labels = xtick_labels_dict[perturb]

    ax = axes[i]

    if perturb == "contrast_increase":
        # Use custom x-axis values for non-uniform contrast increase levels
        x_vals = xtick_labels
        ax.plot(x_vals, dice_scores, marker='o')

        # Display every other label for readability
        display_labels = [f"{val:.2f}" if i % 2 == 0 else "" for i, val in enumerate(xtick_labels)]
        ax.set_xticks(x_vals)
        ax.set_xticklabels(display_labels, fontsize=tick_font, rotation=45)
    else:
        # Use uniform index-based x values for other perturbations
        x_vals = list(range(len(dice_scores)))
        ax.plot(x_vals, dice_scores, marker='o')
        ax.set_xticks(x_vals)
        ax.set_xticklabels(xtick_labels, fontsize=tick_font, rotation=30)

    # Set plot title and axis labels
    ax.set_title(f"{perturb.replace('_', ' ').title()}", fontsize=title_font)
    ax.set_xlabel("Perturbation Level", fontsize=label_font)
    ax.set_ylabel("Mean Dice Score", fontsize=label_font)
    ax.tick_params(axis='y', labelsize=tick_font)
    ax.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "robustness_all_subplot.png"), dpi=300)
plt.close()