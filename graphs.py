import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Back translation Heatmap
def plot_bt_heatmap():
    csv_path = "data/translation_test/back_translation.csv"
    df = pd.read_csv(csv_path, sep=";", index_col="model")

    # Remove avg_score if present
    if "avg_score" in df.columns:
        df = df.drop(columns=["avg_score"])

    # Convert to numeric (in case some values are read as strings)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Plot heatmap
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white"
    )

    plt.xlabel("Language")
    plt.ylabel("Model")
    plt.title("Models Performance in Back Translation Test")
    plt.xticks(rotation=40, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig("graphs/back_translation.png")
    plt.show()

#Back translation Heatmap
plot_bt_heatmap()