from lib.constants import *
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
    matrix = np.full((2, n, 3), 1.0)  # Initialize all cells as white (R=G=B=1)

    # Define RGB colors
    green = [0.0, 0.8, 0.0]
    red = [0.9, 0.1, 0.1]
    yellow = [1.0, 1.0, 0.0]
    grey = [0.7, 0.7, 0.7]
    white = [1.0, 1.0, 1.0]

    for j in range(n):
        # Lower triangle
        pro_i = data[j]["Question Pro"]
        con_i = data[j]["Question Con"]
        op_i = data[j]["Question Op"]
        if pro_i == "AGREE" and con_i == "DISAGREE":
            color = green
        elif pro_i == "DISAGREE" and con_i == "AGREE":
            color = red
        else:
            color = yellow
        matrix[0, j] = color
        color = white if op_i == "AGREE" else grey
        matrix[1, j] = color

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(matrix, interpolation='none')

    # Ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(["Agreement", "Opinion"])

    # Gridlines
    ax.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax.set_yticks(np.arange(2)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.title("LLM Alignment Heatmap")
    plt.tight_layout()
    output_graph_file_marker = f'graphs/plot_llm_alignment_heatmap/{model_name}'
    os.makedirs(output_graph_file_marker, exist_ok=True)
    plt.savefig(output_graph_file_marker+f'/en_GB_heatmap.png')
    plt.show()


model_name = GPT4_MINI
plot_llm_alignment_heatmap(model_name)