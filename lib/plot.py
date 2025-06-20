from constants import *
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_llm_alignment_heatmap(model_name):
    file_in = f'results_for_analysis/languages_experiments/{model_name}/en_GB_raibow_meter.csv'
    
    with open(file_in, 'r') as file:
        data = json.load(file)
        
    n = len(data)
    labels = [item["Subcategory"] for item in data]

    # Initialize matrix (2 rows: Agreement, Opinion; n columns; 3 for RGB)
    matrix = np.full((2, n, 3), 1.0)  # Default white

    # Define colors
    COLORS = {
        "green": [0.0, 0.8, 0.0],
        "red": [0.9, 0.1, 0.1],
        "yellow": [1.0, 1.0, 0.0],
        "grey": [0.7, 0.7, 0.7],
        "white": [1.0, 1.0, 1.0]
    }

    for j, item in enumerate(data):
        pro, con, op = item["Question Pro"], item["Question Con"], item["Question Op"]

        # Alignment color
        if pro == "AGREE" and con == "DISAGREE":
            matrix[0, j] = COLORS["green"]
        elif pro == "DISAGREE" and con == "AGREE":
            matrix[0, j] = COLORS["red"]
        else:
            matrix[0, j] = COLORS["yellow"]

        # Opinion color
        matrix[1, j] = COLORS["white"] if op == "AGREE" else COLORS["grey"]

    # Dynamic figure size
    fig_width = max(12, 0.4 * n)  # Minimum width 12
    fig, ax = plt.subplots(figsize=(fig_width, 5))  # Height fixed to 2.5

    ax.imshow(matrix, interpolation='none', aspect='auto')

    # Axis setup
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Agreement", "Opinion"], fontsize=10)

    # Gridlines
    ax.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax.set_yticks(np.arange(3)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Layout and title
    plt.title(f"LLM Alignment Heatmap – {model_name}", fontsize=12)
    plt.tight_layout(pad=2)

    # Save
    output_dir = f'graphs/plot_llm_alignment_heatmap/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/en_GB_heatmap.png', dpi=300)
    #plt.show()


model_name = GEMMA3_27B


plot_llm_alignment_heatmap(model_name)