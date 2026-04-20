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
    return None

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

##Generate Coherence-validity Heatmap dataframes and heatmaps
def coh_val_heatmaps():
    for scenario in SCENARIOS:
        root_dir = f"{RAINBOW_METER_RESULT_PATH}/{scenario}"
        fact_data = {}
        stance_data = {}

        for model_name in os.listdir(root_dir):
            model_path = os.path.join(root_dir, model_name)
            
            if not os.path.isdir(model_path):
                continue
            model_label = MODELS_LABELS[model_name]
            fact_data[model_label] = {}
            stance_data[model_label] = {}

            for file in os.listdir(model_path):
                # extract language (e.g., az from rm_answers_az.csv)
                
                label = file.replace("rm_answers_", "").replace(".csv", "")
                if scenario == SCENARIO_LANGUAGE:
                    language, country_name = get_lang_from_lang_code(label)
                    label = language
                elif scenario == SCENARIO_NATIONALITY:
                    language, country_name = get_lang_from_lang_code("", label)
                    label = country_name
                elif scenario == SCENARIO_LAN_NAT:
                    language, country_name = get_lang_from_lang_code(label.split("_")[0], label.split("_")[1])
                    label = f"{language} - {country_name}" 
                
                file_path = os.path.join(model_path, file)
                df = pd.read_csv(file_path, sep=";")

                # compute means
                fact_mean = df["Fact Weight coherence by validity"].mean()
                stance_mean = df["Stance Weight coherence by validity"].mean()

                fact_data[model_label][label] = fact_mean
                stance_data[model_label][label] = stance_mean

        # convert to DataFrames
        fact_df = pd.DataFrame.from_dict(fact_data, orient="index")
        stance_df = pd.DataFrame.from_dict(stance_data, orient="index")

        # optional: sort columns alphabetically
        fact_df = fact_df.sort_index(axis=1)
        stance_df = stance_df.sort_index(axis=1)

        # Convert to numeric (in case some values are read as strings)
        fact_df = fact_df.apply(pd.to_numeric, errors="coerce")
        stance_df = stance_df.apply(pd.to_numeric, errors="coerce")

        df.to_csv(f"{TABLES_PATH}/{scenario}/fact_coh_val.csv")
        df.to_csv(f"{TABLES_PATH}/{scenario}/stance_coh_val.csv")
        generate_heatmap(fact_df, "Language", "Model", "Models Fact Weight coherence by validity scores", f"{GRAPHS_PATH}/{scenario}/fact_coh_val.png")
        generate_heatmap(stance_df, "Language", "Model", "Models Stance Weight coherence by validity scores", f"{GRAPHS_PATH}/{scenario}/stance_coh_val.png")

#Back translation Heatmap
back_translation_heatmap()

#Weight coherence by validity scores
coh_val_heatmaps()
