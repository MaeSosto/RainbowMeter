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
        fact_coh_data = {}
        fact_val_data = {}
        fact_coh_val_data = {}
        stance_coh_data = {}
        stance_val_data = {}
        stance_coh_val_data = {}
            
        for model_name in MODEL_LIST:
            model_path = os.path.join(root_dir, model_name)
            model_label = MODEL_LABEL[model_name]
            
            if not os.path.exists(model_path):
                continue
            
            fact_coh_data[model_label] = {}
            fact_val_data[model_label] = {}
            fact_coh_val_data[model_label] = {}
            stance_coh_data[model_label] = {}
            stance_val_data[model_label] = {}
            stance_coh_val_data[model_label] = {}
            
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
                fact_coh_mean = df["Fact Coherence"].mean()
                fact_val_mean = df["Fact Validity"].mean()
                fact_coh_val_mean = df["Fact Weight coherence by validity"].mean()
                stance_coh_mean = df["Stance Coherence"].mean()
                stance_val_mean = df["Stance Validity"].mean()
                stance_mean = df["Stance Weight coherence by validity"].mean()

                fact_coh_data[model_label][label] = fact_coh_mean
                fact_val_data[model_label][label] = fact_val_mean
                fact_coh_val_data[model_label][label] = fact_coh_val_mean
                stance_coh_data[model_label][label] = stance_coh_mean
                stance_val_data[model_label][label] = stance_val_mean
                stance_coh_val_data[model_label][label] = stance_mean

        # convert to DataFrames
        fact_coh_df = pd.DataFrame.from_dict(fact_coh_data, orient="index")
        fact_val_df = pd.DataFrame.from_dict(fact_val_data, orient="index")
        fact_df = pd.DataFrame.from_dict(fact_coh_val_data, orient="index")
        stance_coh_df = pd.DataFrame.from_dict(stance_coh_data, orient="index")
        stance_val_df = pd.DataFrame.from_dict(stance_val_data, orient="index")
        stance_df = pd.DataFrame.from_dict(stance_coh_val_data, orient="index")

        # optional: sort columns alphabetically
        #fact_df = fact_df.sort_index(axis=0)
        fact_coh_df = fact_coh_df.sort_index(axis=1)
        fact_val_df = fact_val_df.sort_index(axis=1)
        fact_df = fact_df.sort_index(axis=1)
        #stance_df = stance_df.sort_index(axis=0)
        stance_coh_df = stance_coh_df.sort_index(axis=1)
        stance_val_df = stance_val_df.sort_index(axis=1)
        stance_df = stance_df.sort_index(axis=1)

        # Convert to numeric (in case some values are read as strings)
        fact_coh_df = fact_coh_df.apply(pd.to_numeric, errors="coerce")
        fact_val_df = fact_val_df.apply(pd.to_numeric, errors="coerce")
        fact_df = fact_df.apply(pd.to_numeric, errors="coerce")
        stance_coh_df = stance_coh_df.apply(pd.to_numeric, errors="coerce")
        stance_val_df = stance_val_df.apply(pd.to_numeric, errors="coerce")
        stance_df = stance_df.apply(pd.to_numeric, errors="coerce")

        fact_coh_df.to_csv(f"{TABLES_PATH}/{scenario}/fact_coh.csv")
        fact_val_df.to_csv(f"{TABLES_PATH}/{scenario}/fact_val.csv")
        fact_df.to_csv(f"{TABLES_PATH}/{scenario}/fact_coh_val.csv")
        stance_coh_df.to_csv(f"{TABLES_PATH}/{scenario}/stance_coh.csv")
        stance_val_df.to_csv(f"{TABLES_PATH}/{scenario}/stance_val.csv")
        stance_df.to_csv(f"{TABLES_PATH}/{scenario}/stance_coh_val.csv")
        generate_heatmap(fact_coh_df, "Language", "Model", "Models Fact Coherence scores", f"{GRAPHS_PATH}/{scenario}/fact_coh.png")
        generate_heatmap(fact_val_df, "Language", "Model", "Models Fact Validity scores", f"{GRAPHS_PATH}/{scenario}/fact_val.png")
        generate_heatmap(fact_df, "Language", "Model", "Models Fact Weight coherence by validity scores", f"{GRAPHS_PATH}/{scenario}/fact_coh_val.png")
        generate_heatmap(stance_coh_df, "Language", "Model", "Models Stance Coherence scores", f"{GRAPHS_PATH}/{scenario}/stance_coh.png")
        generate_heatmap(stance_val_df, "Language", "Model", "Models Stance Validity scores", f"{GRAPHS_PATH}/{scenario}/stance_val.png")
        generate_heatmap(stance_df, "Language", "Model", "Models Stance Weight coherence by validity scores", f"{GRAPHS_PATH}/{scenario}/stance_coh_val.png")

#Back translation Heatmap
back_translation_heatmap()

#Weight coherence by validity scores
coh_val_heatmaps()
