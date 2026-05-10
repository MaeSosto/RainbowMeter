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
      
#Back translation Heatmap
#back_translation_heatmap()

#Weight coherence by validity scores
model_performances()

#Generate the Fact and Stance heatmaps of the MAEs errors of all the models   
#maes()
