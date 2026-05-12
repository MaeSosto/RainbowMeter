from constants import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from  matplotlib.colors import LinearSegmentedColormap

CMAP_RG=LinearSegmentedColormap.from_list('rg',["r", "y", "g"], N=256) 
CMAP_RG_INVERTED=LinearSegmentedColormap.from_list('rg',["g", "y", "r"], N=256) 

#Generate, show and save an heatmap
def generate_heatmap(df, xlabel, ylabel, title, savefig, annot = False, cmap = CMAP_RG):
    #df = df.sort_index(axis=1)
    
    # Plot heatmap
    plt.figure(figsize=(16, 6))
    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        annot=annot,
        annot_kws={"fontsize":8},
        fmt=".1f"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=40, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(savefig)
    print(f"Saved: {savefig}")
    #plt.show()
    
#Generate back translation dataframe and heatmap
def back_translation():
    df = pd.read_csv("data/translation_test/back_translation.csv", sep=";", index_col="model")

    # Remove avg_score if present
    if "avg_score" in df.columns:
        df = df.drop(columns=["avg_score"])

    # Convert to numeric (in case some values are read as strings)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    plt.figure(figsize=(16, 6))
    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=CMAP_RG,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        annot_kws={"fontsize":8},
        fmt=".1f"
    )

    plt.xlabel("Language")
    plt.ylabel("Model")
    plt.title("Models Performance in Back Translation Test")
    plt.xticks(rotation=40, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_PATH}/back_translation.png")
    print(f"Saved: {f"{GRAPHS_PATH}/back_translation.png"}")

def model_performances():

    metrics = {
        "fact_coh": "Fact Coherence",
        "fact_val": "Fact Validity",
        "fact_coh_val": "Fact Weight coherence by validity",
        "stance_coh": "Stance Coherence",
        "stance_val": "Stance Validity",
        "stance_coh_val": "Stance Weight coherence by validity",
    }

    title_map = {
        "fact_coh": "Models Fact Coherence scores",
        "fact_val": "Models Fact Validity scores",
        "fact_coh_val": "Models Fact Weight coherence by validity scores",
        "stance_coh": "Models Stance Coherence scores",
        "stance_val": "Models Stance Validity scores",
        "stance_coh_val": "Models Stance Weight coherence by validity scores",
    }

    for scenario in SCENARIOS:

        for m in metrics.keys():

            file_path = (
                f"{EVALUATIONS_PATH}/"
                f"{MODELS_PERFORMANCES_PATH}/"
                f"{scenario}/{m}.csv"
            )

            if not os.path.exists(file_path):
                continue

            # IMPORTANT: first column is the model name index
            df_metric = pd.read_csv(file_path, sep=",", index_col=0)

            # Convert only data columns to numeric
            df_metric = df_metric.apply(pd.to_numeric, errors="coerce")

            # Keep original model order from CSV
            # Sort only languages if desired
            df_metric = df_metric.sort_index(axis=1)

            generate_heatmap(
                df=df_metric,
                xlabel="Language",
                ylabel="Model",
                title=f"{title_map[m]} in {scenario} Scenario",
                savefig=f"{GRAPHS_PATH}/{MODELS_PERFORMANCES_PATH}/{scenario}/{m}.png",
                annot=True
            )

#Calculate the MAEs errors            
def mae_models():
    for test in [FACT, STANCE]:
        csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

        if not os.path.exists(csv_path):
            print(f"Missing file: {csv_path}")
            return

        df_scenario = pd.read_csv(csv_path, sep=";")

        # Iterate over scenarios contained in the dataframe
        for scenario in SCENARIOS:
            df = df_scenario[df_scenario[SCENARIO] == scenario]
            
            df["Language - Country"] = (df["languages"].astype(str)+ " | "+ df["Country"].astype(str))
            df["Model"] = df["Model"].astype(str).map(MODEL_LABEL)
            ordered_labels = [
                MODEL_LABEL.get(model, model)
                for model in MODEL_LIST
            ]

            df["Model"] = pd.Categorical(
                df["Model"],
                categories=ordered_labels,
                ordered=True,
            )

            df = df.sort_values("Model")

            # Pivot tables
            fact_pivot = df.pivot_table(
                index="Model",
                columns="Language - Country",
                values=f"{test} MAE"
            )

            generate_heatmap(
                df = fact_pivot,
                xlabel = "Language and Country",
                ylabel = "Model",
                title = f"{test} MAE Heatmap",
                savefig = f"{GRAPHS_PATH}/MAE/{scenario}/{test}_models.png",
                cmap= CMAP_RG_INVERTED
            )
              
def line_graphs_percentage_plain():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df_scenario = pd.read_csv(csv_path, sep=";")

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        df = df_scenario[df_scenario[SCENARIO] == scenario]

        if scenario == SCENARIO_LANGUAGE:
            df["x_label"] = df["languages"]
        elif scenario == SCENARIO_NATIONALITY:
            df["x_label"] = df["Country"]
        elif scenario == SCENARIO_LAN_NAT:
            df["x_label"] = (df["languages"].astype(str) + " | "+ df["Country"].astype(str))

        df = df.sort_values("x_label")
        models = df["Model"].unique()

        for model in models:
            df_model = (df[df["Model"] == model].copy().reset_index(drop=True))

            x = range(len(df_model))
            plt.figure(figsize=(26, 10))

            # Rainbow Map
            plt.plot(
                x,
                df_model[RAINBOW_MAP],
                linestyle="--",
                linewidth=2,
                label=RAINBOW_MAP
            )
            
            for test in [FACT, STANCE]:
                plt.plot(
                    x,
                    df_model[test],
                    marker="o",
                    label="Fact Prediction"
                )

                # Emphasize distance
                plt.fill_between(
                    x,
                    df_model[test],
                    df_model[RAINBOW_MAP],
                    alpha=0.15
                )

            plt.xticks(
                x,
                df_model["x_label"],
                rotation=90
            )

            plt.ylabel("Score in %")
            plt.xlabel("Language / Country")

            plt.title(f"{model} | {SCENARIO_LABELS[scenario]} | Scores in %")

            plt.legend()
            plt.tight_layout()

            save_path = (f"{GRAPHS_PATH}/percentage_plain/{scenario}/{model}.png")
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.close()

def mae_aggregated_language_nat():
    csv_path = f"{EVALUATIONS_PATH}/lang_country_mae_summary.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:
        language_col = f"{SCENARIO_LANGUAGE}_{test}_MAE"
        nationality_col = f"{SCENARIO_NATIONALITY}_{test}_MAE"
        lang_nat_col = f"{SCENARIO_LAN_NAT}_{test}_MAE"

        language_rows = (
            df[df[language_col].notna()][["Key", language_col]]
            .copy()
            .rename(columns={
                "Key": "Language",
                language_col: "Value"
            })
        )

        country_rows = (
            df[df[nationality_col].notna()][["Key", nationality_col]]
            .copy()
            .rename(columns={
                "Key": "Country",
                nationality_col: "Value"
            })
        )

        lang_country_rows = (
            df[df[lang_nat_col].notna()][["Key", lang_nat_col]]
            .copy()
        )

        lang_country_rows[["Language", "Country"]] = (
            lang_country_rows["Key"]
            .str.split(r"\s*\|\s*", expand=True)
        )

        lang_country_rows.rename(columns={
            lang_nat_col: "Value"
        }, inplace=True)

        languages = sorted(
            set(language_rows["Language"].dropna())
            | set(lang_country_rows["Language"].dropna())
        )

        countries = sorted(
            set(country_rows["Country"].dropna())
            | set(lang_country_rows["Country"].dropna())
        )

        full_rows =  languages + [SCENARIO_LABELS[SCENARIO_NATIONALITY]]
        full_columns = [SCENARIO_LABELS[SCENARIO_LANGUAGE]] + countries

        matrix = pd.DataFrame(
            np.nan,
            index=full_rows,
            columns=full_columns,
        )

        for _, row in language_rows.iterrows():

            matrix.loc[
                row["Language"],
                SCENARIO_LABELS[SCENARIO_LANGUAGE]
            ] = row["Value"]

        for _, row in country_rows.iterrows():

            matrix.loc[
                SCENARIO_LABELS[SCENARIO_NATIONALITY],
                row["Country"]
            ] = row["Value"]

        for _, row in lang_country_rows.iterrows():

            matrix.loc[
                row["Language"],
                row["Country"]
            ] = row["Value"]

        plt.figure(
            figsize=(
                max(14, len(matrix.columns) * 0.45),
                max(12, len(matrix.index) * 0.35),
            )
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="grey",
            square=False,
            cbar_kws={"label": "MAE"},
        )

        plt.title(f"{test} MAE Heatmap")
        plt.xlabel("Countries")
        plt.ylabel("Languages")
        plt.xticks(rotation=40, ha='right', fontsize=15)
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_png = (f"{GRAPHS_PATH}/MAE/{test}_mae.png")
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_png}")
        
def line_graphs_percentage_difference():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df_scenario = pd.read_csv(csv_path, sep=";")

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        df = df_scenario[df_scenario[SCENARIO] == scenario]

        if scenario == SCENARIO_LANGUAGE:
            df["x_label"] = df["languages"]
        elif scenario == SCENARIO_NATIONALITY:
            df["x_label"] = df["Country"]
        elif scenario == SCENARIO_LAN_NAT:
            df["x_label"] = (df["languages"].astype(str) + " | "+ df["Country"].astype(str))

        df = df.sort_values("x_label")
        models = df["Model"].unique()

        for model in models:
            df_model = (df[df["Model"] == model].copy().reset_index(drop=True))

            x = range(len(df_model))
            plt.figure(figsize=(26, 10))
            
            for test in [FACT, STANCE]:
                df_model[f"{test} Percentage Difference" ] = (df_model[test] - df_model[RAINBOW_MAP])

                plt.plot(
                    x,
                    df_model[f"{test} Percentage Difference"],
                    marker="o",
                    label=f"{test} Percentage Difference" 
                )

                plt.fill_between(
                    x,
                    0,
                    df_model[f"{test} Percentage Difference" ],
                    alpha=0.15
                )

            # perfect alignment
            plt.axhline(
                y=0,
                linestyle="--",
                label="Perfect Alignment"
            )

            plt.xticks(
                x,
                df_model["x_label"],
                rotation=90
            )

            plt.ylabel("Prediction - Rainbow Map")
            plt.xlabel("Language / Country")
            plt.title(f"{model} | {scenario} | Score Difference in %")

            plt.legend()
            plt.tight_layout()

            save_path = (f"{GRAPHS_PATH}/percentage_difference/{scenario}/{model}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.close()

def line_graphs_pvalue():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df_scenario = pd.read_csv(csv_path, sep=";")

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        df = df_scenario[df_scenario[SCENARIO] == scenario]

        if scenario == SCENARIO_LANGUAGE:
            df["x_label"] = df["languages"]
        elif scenario == SCENARIO_NATIONALITY:
            df["x_label"] = df["Country"]
        elif scenario == SCENARIO_LAN_NAT:
            df["x_label"] = (df["languages"].astype(str) + " | "+ df["Country"].astype(str))

        df = df.sort_values("x_label")
        models = df["Model"].unique()

        for model in models:
            df_model = (df[df["Model"] == model].copy().reset_index(drop=True))

            x = range(len(df_model))
            plt.figure(figsize=(26, 10))

            for test in [FACT, STANCE]:
                plt.plot(
                    x,
                    df_model[f"{test} Pvalue"],
                    marker="o",
                    label=f"{test} P-Value"
                )

            # significance threshold
            plt.axhline(
                y=0.05,
                linestyle="--",
                label="p = 0.05"
            )

            plt.xticks(x, df_model["x_label"], rotation=90)
            plt.ylabel("P-Value")
            plt.xlabel("Language / Country")
            plt.title(f"{model} | {scenario} | P-Value Comparison")
            plt.legend()
            plt.tight_layout()

            save_path = (f"{GRAPHS_PATH}/pvalues/{scenario}/{model}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.close()
        
        
#Back translation Heatmap
#back_translation()

#Weight coherence by validity scores
model_performances()

#Generate the Fact and Stance heatmaps of the MAEs errors of all the models   
mae_models()
#mae_aggregated_language_nat()

#Create the line graphs 
#line_graphs_percentage_plain()
#line_graphs_percentage_difference()
#line_graphs_pvalue()

print(f"✅ Graphs Generated")