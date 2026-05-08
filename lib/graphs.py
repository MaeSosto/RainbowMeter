from models import *
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
def generate_heatmap(df, xlabel, ylabel, title, savefig):
    # Plot heatmap
    plt.figure(figsize=(18, 6))
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

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=40, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(savefig)
    #plt.show()
    
#Generate back translation dataframe and heatmap
def back_translation_heatmap():
    csv_path = "data/translation_test/back_translation.csv"
    df = pd.read_csv(csv_path, sep=";", index_col="model")

    # Remove avg_score if present
    if "avg_score" in df.columns:
        df = df.drop(columns=["avg_score"])

    # Convert to numeric (in case some values are read as strings)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    generate_heatmap(df, "Language", "Model", "Models Performance in Back Translation Test", f"{GRAPHS_PATH}/back_translation.png")
    return df

def coh_val_heatmaps():
    # Define metrics once
    metrics = {
        "fact_coh": "Fact Coherence",
        "fact_val": "Fact Validity",
        "fact_coh_val": "Fact Weight coherence by validity",
        "stance_coh": "Stance Coherence",
        "stance_val": "Stance Validity",
        "stance_coh_val": "Stance Weight coherence by validity",
    }

    for scenario in SCENARIOS:
        root_dir = f"{RAINBOW_METER_RESULT_PATH}/{scenario}"

        # metric_name -> {model -> {label -> value}}
        data = {m: {} for m in metrics}

        for model_name in MODEL_LIST:
            model_path = os.path.join(root_dir, model_name)
            if not os.path.exists(model_path):
                continue

            model_label = MODEL_LABEL[model_name]

            # initialize model entries
            for m in metrics:
                data[m][model_label] = {}

            for file in os.listdir(model_path):
                if not file.endswith(".csv"):
                    continue

                raw_label = file.replace("rm_answers_", "").replace(".csv", "")
                label = get_label(raw_label, scenario)

                df = pd.read_csv(os.path.join(model_path, file), sep=";")

                # compute all means in one go
                means = df[list(metrics.values())].mean()

                for m, col in metrics.items():
                    data[m][model_label][label] = means[col]

        # Convert, clean, save, plot
        for m, metric_data in data.items():
            df_metric = pd.DataFrame.from_dict(metric_data, orient="index")

            df_metric = (
                df_metric
                .sort_index(axis=1)
                .apply(pd.to_numeric, errors="coerce")
            )

            # Save CSV
            out_csv = f"{TABLES_PATH}/{MODELS_STATS_PATH}/{scenario}/{m}.csv"
            df_metric.to_csv(out_csv)

            # Pretty title
            title_map = {
                "fact_coh": "Models Fact Coherence scores",
                "fact_val": "Models Fact Validity scores",
                "fact_coh_val": "Models Fact Weight coherence by validity scores",
                "stance_coh": "Models Stance Coherence scores",
                "stance_val": "Models Stance Validity scores",
                "stance_coh_val": "Models Stance Weight coherence by validity scores",
            }

            out_png = f"{GRAPHS_PATH}/{MODELS_STATS_PATH}/{scenario}/{m}.png"

            generate_heatmap(
                df_metric,
                "Language",
                "Model",
                title_map[m] + f" in {scenario} Scenario",
                out_png
            )
            
#Back translation Heatmap
#back_translation_heatmap()

#Weight coherence by validity scores
coh_val_heatmaps()
