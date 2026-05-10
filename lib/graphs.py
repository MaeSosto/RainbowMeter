from constants import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from  matplotlib.colors import LinearSegmentedColormap

CMAP_RG=LinearSegmentedColormap.from_list('rg',["r", "y", "g"], N=256) 

#Given a lan id, return the language name
def get_lang_from_lang_code(lang_code = "", count_code = ""):
    for country_name in COUNTRIES_FILE: 
        for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
            if lang_code == COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num] and (count_code == "" or count_code == COUNTRIES_FILE[country_name][ID]):
                return language, country_name
            if count_code == COUNTRIES_FILE[country_name][ID] and lang_code == "":
                return language, country_name
    return None, None

#Given a scenario and the name of the rainbow meter file, it returns the language/country or both as label for the graph
def get_label(label, scenario):
    if scenario == SCENARIO_LANGUAGE:
        language, _ = get_lang_from_lang_code(label)
        return language
    elif scenario == SCENARIO_NATIONALITY:
        _, country = get_lang_from_lang_code("", label)
        return country
    elif scenario == SCENARIO_LAN_NAT:
        lang_code, country_code = label.split("_")
        language, country = get_lang_from_lang_code(lang_code, country_code)
        return f"{language} - {country}"
    return label

#Generate, show and save an heatmap
def generate_heatmap(df, xlabel, ylabel, title, savefig, annot = False):
    #df = df.sort_index(axis=1)
    
    # Plot heatmap
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=CMAP_RG,
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
    #plt.show()
    
#Generate back translation dataframe and heatmap
def back_translation():
    df = pd.read_csv("data/translation_test/back_translation.csv", sep=";", index_col="model")

    # Remove avg_score if present
    if "avg_score" in df.columns:
        df = df.drop(columns=["avg_score"])

    # Convert to numeric (in case some values are read as strings)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    generate_heatmap(df, "Language", "Model", "Models Performance in Back Translation Test", f"{GRAPHS_PATH}/back_translation.png")

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

        output_dir = f"{GRAPHS_PATH}/{MODELS_PERFORMANCES_PATH}/{scenario}"
        os.makedirs(output_dir, exist_ok=True)

        for m in metrics.keys():

            file_path = (
                f"{EVALUATIONS_PATH}/"
                f"{MODELS_PERFORMANCES_PATH}/"
                f"{scenario}/{m}.csv"
            )

            if not os.path.exists(file_path):
                continue

            # IMPORTANT: first column is the model name index
            df_metric = pd.read_csv(
                file_path,
                sep=",",
                index_col=0
            )

            # Convert only data columns to numeric
            df_metric = df_metric.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # Keep original model order from CSV
            # Sort only languages if desired
            df_metric = df_metric.sort_index(axis=1)

            save_path = f"{output_dir}/{m}.png"

            generate_heatmap(
                df=df_metric,
                xlabel="Language",
                ylabel="Model",
                title=f"{title_map[m]} in {scenario} Scenario",
                savefig=save_path,
                annot=True
            )

            print(f"Saved: {save_path}")

#Calculate the MAEs errors            
def maes():
    for test in [FACT, STANCE]:
        for scenario in SCENARIOS:
            df = pd.read_csv(f"{EVALUATIONS_PATH}/{scenario}/general_stats.csv", sep=";")
            
            df["Language - Country"] = (df["languages"].astype(str)+ " | "+ df["Country"].astype(str))

            # Pivot tables
            fact_pivot = df.pivot_table(
                index="Model",
                columns="Language - Country",
                values=f"{test} MAE"
            )

            #stance_pivot = stance_pivot.sort_index()
            generate_heatmap(
                df = fact_pivot,
                xlabel = "Language and Country",
                ylabel = "Model",
                title = f"{test} MAE Heatmap",
                savefig = f"{GRAPHS_PATH}/{scenario}/{test}_mae.png"
            )

def generate_line_graphs():

    tests = {
        FACT: {
            "score": "Fact",
            "pvalue": "Fact Pvalue",
            "mae": "Fact MAE",
        },
        STANCE: {
            "score": "Stance",
            "pvalue": "Stance Pvalue",
            "mae": "Stance MAE",
        }
    }

    for test, cols in tests.items():

        score_col = cols["score"]
        pvalue_col = cols["pvalue"]
        mae_col = cols["mae"]

        for scenario in SCENARIOS:

            df = pd.read_csv(
                f"{EVALUATIONS_PATH}/{scenario}/general_stats.csv",
                sep=";"
            )

            # ==========================================================
            # X AXIS
            # ==========================================================

            if scenario == SCENARIO_LANGUAGE:

                df["x_label"] = df["languages"]

            elif scenario == SCENARIO_NATIONALITY:

                df["x_label"] = df["Country"]

            elif scenario == SCENARIO_LAN_NAT:

                df["x_label"] = (
                    df["languages"].astype(str)
                    + " | "
                    + df["Country"].astype(str)
                )

            # ==========================================================
            # SORTING
            # ==========================================================

            df = df.sort_values("Rainbow Map")

            models = df["Model"].unique()

            # ==========================================================
            # GENERATE PER MODEL
            # ==========================================================

            for model in models:

                df_model = (
                    df[df["Model"] == model]
                    .copy()
                    .reset_index(drop=True)
                )

                # ======================================================
                # SIGNED ERROR
                # ======================================================

                df_model["Signed Error"] = (
                    df_model[score_col]
                    - df_model["Rainbow Map"]
                )

                x = range(len(df_model))

                # ======================================================
                # 1. PREDICTION VS REALITY
                # ======================================================

                plt.figure(figsize=(24, 8))

                plt.plot(
                    x,
                    df_model[score_col],
                    marker="o",
                    label=f"{score_col} Prediction"
                )

                plt.plot(
                    x,
                    df_model["Rainbow Map"],
                    linestyle="--",
                    label="Rainbow Map"
                )

                # emphasize distance
                plt.fill_between(
                    x,
                    df_model[score_col],
                    df_model["Rainbow Map"],
                    alpha=0.2
                )

                plt.xticks(
                    x,
                    df_model["x_label"],
                    rotation=90
                )

                plt.ylabel("Score")
                plt.xlabel("Language / Country")

                plt.title(
                    f"{model} | {scenario} | "
                    f"{test} Prediction vs Rainbow Map"
                )

                plt.legend()
                plt.tight_layout()

                save_path = (
                    f"{GRAPHS_PATH}/linegraphs/"
                    f"prediction_vs_real/"
                    f"{scenario}/{test}/{model}.png"
                )

                os.makedirs(
                    os.path.dirname(save_path),
                    exist_ok=True
                )

                plt.savefig(save_path)
                plt.close()

                # ======================================================
                # 2. PVALUE GRAPH
                # ======================================================

                plt.figure(figsize=(24, 8))

                plt.plot(
                    x,
                    df_model[pvalue_col],
                    marker="o",
                    label=pvalue_col
                )

                # significance threshold
                plt.axhline(
                    y=0.05,
                    linestyle="--",
                    label="p = 0.05"
                )

                plt.xticks(
                    x,
                    df_model["x_label"],
                    rotation=90
                )

                plt.ylabel("P-Value")
                plt.xlabel("Language / Country")

                plt.title(
                    f"{model} | {scenario} | "
                    f"{test} P-Values"
                )

                plt.legend()
                plt.tight_layout()

                save_path = (
                    f"{GRAPHS_PATH}/linegraphs/"
                    f"pvalues/"
                    f"{scenario}/{test}/{model}.png"
                )

                os.makedirs(
                    os.path.dirname(save_path),
                    exist_ok=True
                )

                plt.savefig(save_path)
                plt.close()

                # ======================================================
                # 3. SIGNED ERROR GRAPH
                # ======================================================

                plt.figure(figsize=(24, 8))

                plt.plot(
                    x,
                    df_model["Signed Error"],
                    marker="o",
                    label="Signed Error"
                )

                # perfect alignment
                plt.axhline(
                    y=0,
                    linestyle="--",
                    label="Perfect Alignment"
                )

                # emphasize magnitude of error
                plt.fill_between(
                    x,
                    0,
                    df_model["Signed Error"],
                    alpha=0.2
                )

                plt.xticks(
                    x,
                    df_model["x_label"],
                    rotation=90
                )

                plt.ylabel(
                    "Prediction - Rainbow Map"
                )

                plt.xlabel("Language / Country")

                plt.title(
                    f"{model} | {scenario} | "
                    f"{test} Signed Error"
                )

                plt.legend()
                plt.tight_layout()

                save_path = (
                    f"{GRAPHS_PATH}/linegraphs/"
                    f"signed_error/"
                    f"{scenario}/{test}/{model}.png"
                )

                os.makedirs(
                    os.path.dirname(save_path),
                    exist_ok=True
                )

                plt.savefig(save_path)
                plt.close()
              
def generate_combined_line_graphs():

    for scenario in SCENARIOS:

        df = pd.read_csv(
            f"{EVALUATIONS_PATH}/{scenario}/general_stats.csv",
            sep=";"
        )

        # ==============================================================
        # X AXIS LABELS
        # ==============================================================

        if scenario == SCENARIO_LANGUAGE:

            df["x_label"] = df["languages"]

        elif scenario == SCENARIO_NATIONALITY:

            df["x_label"] = df["Country"]

        elif scenario == SCENARIO_LAN_NAT:

            df["x_label"] = (
                df["languages"].astype(str)
                + " | "
                + df["Country"].astype(str)
            )

        # ==============================================================
        # SORT FOR CONSISTENT VISUALIZATION
        # ==============================================================

        df = df.sort_values("Rainbow Map")

        models = df["Model"].unique()

        # ==============================================================
        # GENERATE PER MODEL
        # ==============================================================

        for model in models:

            df_model = (
                df[df["Model"] == model]
                .copy()
                .reset_index(drop=True)
            )

            # ==========================================================
            # AGGREGATED ERROR
            # ==============================================================

            df_model["Fact Signed Error"] = (
                df_model["Fact"]
                - df_model["Rainbow Map"]
            )

            df_model["Stance Signed Error"] = (
                df_model["Stance"]
                - df_model["Rainbow Map"]
            )

            x = range(len(df_model))

            # ==========================================================
            # 1. FACT + STANCE VS RAINBOW MAP
            # ==============================================================

            plt.figure(figsize=(26, 10))

            # Fact
            plt.plot(
                x,
                df_model["Fact"],
                marker="o",
                label="Fact Prediction"
            )

            # Stance
            plt.plot(
                x,
                df_model["Stance"],
                marker="o",
                label="Stance Prediction"
            )

            # Rainbow Map
            plt.plot(
                x,
                df_model["Rainbow Map"],
                linestyle="--",
                linewidth=2,
                label="Rainbow Map"
            )

            # Emphasize distance
            plt.fill_between(
                x,
                df_model["Fact"],
                df_model["Rainbow Map"],
                alpha=0.15
            )

            plt.fill_between(
                x,
                df_model["Stance"],
                df_model["Rainbow Map"],
                alpha=0.15
            )

            plt.xticks(
                x,
                df_model["x_label"],
                rotation=90
            )

            plt.ylabel("Score")
            plt.xlabel("Language / Country")

            plt.title(
                f"{model} | {scenario} | "
                f"Predictions vs Rainbow Map"
            )

            plt.legend()
            plt.tight_layout()

            save_path = (
                f"{GRAPHS_PATH}/linegraphs/"
                f"combined_predictions/"
                f"{scenario}/{model}.png"
            )

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            plt.savefig(save_path)
            plt.close()

            # ==========================================================
            # 2. PVALUE COMPARISON
            # ==============================================================

            plt.figure(figsize=(26, 10))

            plt.plot(
                x,
                df_model["Fact Pvalue"],
                marker="o",
                label="Fact P-Value"
            )

            plt.plot(
                x,
                df_model["Stance Pvalue"],
                marker="o",
                label="Stance P-Value"
            )

            # significance threshold
            plt.axhline(
                y=0.05,
                linestyle="--",
                label="p = 0.05"
            )

            plt.xticks(
                x,
                df_model["x_label"],
                rotation=90
            )

            plt.ylabel("P-Value")
            plt.xlabel("Language / Country")

            plt.title(
                f"{model} | {scenario} | "
                f"P-Value Comparison"
            )

            plt.legend()
            plt.tight_layout()

            save_path = (
                f"{GRAPHS_PATH}/linegraphs/"
                f"combined_pvalues/"
                f"{scenario}/{model}.png"
            )

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            plt.savefig(save_path)
            plt.close()

            # ==========================================================
            # 3. SIGNED ERROR COMPARISON
            # ==============================================================

            plt.figure(figsize=(26, 10))

            plt.plot(
                x,
                df_model["Fact Signed Error"],
                marker="o",
                label="Fact Signed Error"
            )

            plt.plot(
                x,
                df_model["Stance Signed Error"],
                marker="o",
                label="Stance Signed Error"
            )

            # perfect alignment
            plt.axhline(
                y=0,
                linestyle="--",
                label="Perfect Alignment"
            )

            # emphasize magnitude
            plt.fill_between(
                x,
                0,
                df_model["Fact Signed Error"],
                alpha=0.15
            )

            plt.fill_between(
                x,
                0,
                df_model["Stance Signed Error"],
                alpha=0.15
            )

            plt.xticks(
                x,
                df_model["x_label"],
                rotation=90
            )

            plt.ylabel(
                "Prediction - Rainbow Map"
            )

            plt.xlabel("Language / Country")

            plt.title(
                f"{model} | {scenario} | "
                f"Signed Error Comparison"
            )

            plt.legend()
            plt.tight_layout()

            save_path = (
                f"{GRAPHS_PATH}/linegraphs/"
                f"combined_signed_error/"
                f"{scenario}/{model}.png"
            )

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            plt.savefig(save_path)
            plt.close()
              
#Back translation Heatmap
#back_translation_heatmap()

#Weight coherence by validity scores
#model_performances()

#Generate the Fact and Stance heatmaps of the MAEs errors of all the models   
#maes()

#generate_line_graphs()
generate_combined_line_graphs()