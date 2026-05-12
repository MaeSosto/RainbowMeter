# Create data
from constants import *
import scipy.stats as stats
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

WEIGHT_LIST = CRITERIA_WEIGHTS_DF["Weight"].values.astype(float).tolist()

def wilcoxon(group1, group2):
    return round(stats.wilcoxon(group1, group2).statistic.astype(float),2), round(stats.wilcoxon(group1, group2).pvalue.astype(float), 2)

#Get the full list of scores and adjust them accordingly to the Family category contraints
def citeria_adjustment(rm_scores):
    #Exceptions:
    # - “Universality of prohibition of medical interventions” crtierion can only be awarded when “Prohibition of medical intervention before child is able to give informed consent” is awarded.
    rm_scores[61] = rm_scores[61] if rm_scores[60] > 0 else 0
    
    # - “Existence of effective monitoring mechanism” criterion can only be awarded when “Prohibition of medical intervention before child is able to give informed consent” is awarded
    rm_scores[62] = rm_scores[62] if rm_scores[60] > 0 else 0

    # - The criteria Marriage equality, Registered partnership (similar rights to marriage), Registered partnership (limited rights), and Cohabitation, from the “Family” category, reflect different decremental levels of support from legislation on the matter of LGBTI partnership. In the Rainbow Meter, as in the Rainbow Map, we consider a single score for the 4 categories, which corresponds to the score of the satisfied criterion with the highest weight. Where Marriage equality has a higher weight and Cohabitation a lower one.
    family_scores = rm_scores[25:29]
    family_weights = WEIGHT_LIST[25:29]

    max_score = 0
    for idx, sco in enumerate(reversed(family_scores)):
        if sco >= max_score:
            max_score = sco
            selected_weight = family_weights[3-idx]

    adjusted_scores = np.concatenate([
        rm_scores[:25],
        [max_score],
        rm_scores[29:]
    ])

    adjusted_weights = np.concatenate([
        WEIGHT_LIST[:25],
        [selected_weight],
        WEIGHT_LIST[29:]
    ])

    return np.multiply(adjusted_scores, adjusted_weights).astype(float).tolist()

#Creates a table with the model performances in matter of Coherence and Valitidy
def model_performances():

    metrics = {
        "fact_coh": "Fact Coherence",
        "fact_val": "Fact Validity",
        "fact_coh_val": "Fact Weight coherence by validity",
        "stance_coh": "Stance Coherence",
        "stance_val": "Stance Validity",
        "stance_coh_val": "Stance Weight coherence by validity",
    }
    
    #Iterate on the scenario
    for scenario in SCENARIOS: 
        # metric_name -> {model -> {label -> value}}
        data = {m: {} for m in metrics}
        
        #Iterate on every model
        for model_name in MODEL_LIST:
            model_label = MODEL_LABEL[model_name]
            
            for m in metrics:
                data[m][model_label] = {}

            #Iterate on every country
            for country_name, country_data in COUNTRIES_FILE.items(): #tqdm.tqdm(
                country_name = country_name
                country_id = country_data[ID]
            
                #Iterate on every language and citizenship 
                for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                    language = language
                    language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
                
                    #Retrieve the Rainbow Meter of a specific language (if exist)
                    existent_rm_df = get_rainbow_meter_file_answers(
                        scenario = scenario,
                        model_name= model_name,
                        language_code = language_code,
                        country_id = country_id
                    )
                    if existent_rm_df.empty:  #RM doesn't exist or is incomplete
                        continue
                    
                    # compute all means in one go
                    metric_values = list(metrics.values())
                    means = existent_rm_df[metric_values].mean()

                    if scenario == SCENARIO_LANGUAGE:
                        label = language
                    elif scenario == SCENARIO_COUNTRY:
                        label = country_name
                    elif scenario == SCENARIO_LAN_NAT:
                        label = f"{language} - {country_name}" 
                        
                    for m, col in metrics.items():
                        data[m][model_label][label] = round(means[col], 2)

        # Convert, clean, save, plot
        for m, metric_data in data.items():
            df_metric = pd.DataFrame.from_dict(metric_data, orient="index")

            df_metric = (
                df_metric
                .sort_index(axis=1)
                .apply(pd.to_numeric, errors="coerce")
            )

            # Save CSV
            df_metric.to_csv(f"{EVALUATIONS_PATH}/{MODELS_PERFORMANCES_PATH}/{scenario}/{m}.csv")

#Create a table with the grouped scores of MAE and percentages across model-scenario(lan and country) 
def general_stats():
    #Iterate on the scenario
    percentage_results = {
        MODEL: [],
        SCENARIO: [],
        COUNTRY: [],
        LANGUAGES: [],
        FACT: [],
        STANCE: [],
        RAINBOW_MAP: [],
        FACT + " Statistics": [],
        FACT + " Pvalue": [],
        FACT + " MAE": [],
        STANCE + " Statistics": [],
        STANCE + " Pvalue": [],
        STANCE + " MAE": [],
    }
    for scenario in SCENARIOS: #SCENARIOS:
        
        #Iterate on every model
        for model_name in MODEL_LIST:
            
            #Iterate on every country
            for country_name, country_data in COUNTRIES_FILE.items(): #tqdm.tqdm(
                country_name = country_name
                country_id = country_data[ID]
                citizenship = country_data[CITIZENSHIP]
            
                #Iterate on every language and citizenship 
                for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                    language = language
                    language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
                
                    #Retrieve the Rainbow Meter of a specific language (if exist)
                    existent_rm_df = get_rainbow_meter_file_answers(
                        scenario = scenario,
                        model_name= model_name,
                        language_code = language_code,
                        country_id = country_id
                    )
                    if existent_rm_df.empty:  #RM doesn't exist or is incomplete
                        continue
                    
                    percentage_results[MODEL].append(model_name)
                    percentage_results[SCENARIO].append(scenario)
                    percentage_results[LANGUAGES].append(language)
                    percentage_results[COUNTRY].append(country_name)
                    
                    #Get the Rainbow Map real Scores of that country-lan
                    rainbow_map_scores = RAINBOW_MAP_DF.loc[country_id].drop("country_name").drop("Rank").values.astype(float).tolist()
                    rainbow_map_scores = citeria_adjustment(rainbow_map_scores)
                    percentage_results[RAINBOW_MAP].append(round(float(sum(rainbow_map_scores)), 2))
                    
                    #Get the rainbow Meter scores of that country
                    rainbow_meter_fact_scores = existent_rm_df[FACT].values.astype(float).tolist()
                    rainbow_meter_supp_scores = existent_rm_df[SUPPORT].values.astype(float).tolist()
                    rainbow_meter_opp_scores = existent_rm_df[OPPOSITION].values.astype(float).tolist()
                    rainbow_meter_stance_scores = [
                        (s + (1 - o)) / 2
                        for s, o in zip(rainbow_meter_supp_scores, rainbow_meter_opp_scores)
                    ]
                    rainbow_meter_fact_scores = citeria_adjustment(rainbow_meter_fact_scores)
                    rainbow_meter_stance_scores = citeria_adjustment(rainbow_meter_stance_scores)
                    
                    percentage_results[FACT].append(round(sum(rainbow_meter_fact_scores), 2))
                    percentage_results[STANCE].append(round(sum(rainbow_meter_stance_scores), 2))
                    wil1, wil2 = wilcoxon(rainbow_meter_fact_scores, rainbow_map_scores)
                    percentage_results[FACT+" "+ "Statistics"].append(wil1)
                    percentage_results[FACT+" "+ "Pvalue"].append(wil2)
                    wil1, wil2 = wilcoxon(rainbow_meter_stance_scores, rainbow_map_scores)
                    percentage_results[STANCE+" "+ "Statistics"].append(wil1) 
                    percentage_results[STANCE+" "+ "Pvalue"].append(wil2)
                    percentage_results[FACT+" MAE"].append(round(mae(rainbow_map_scores, rainbow_meter_fact_scores), 2))
                    percentage_results[STANCE+" MAE"].append(round(mae(rainbow_map_scores, rainbow_meter_stance_scores), 2))
                        
    #Export Results
    results_df = pd.DataFrame(percentage_results)
    results_df.to_csv(f"{EVALUATIONS_PATH}/general_stats.csv", sep=";", index=False)

#Create a table with the MAEs calculated accross models 
def models_mae():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")
    merged_df = None

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        scenario_df = df[df[SCENARIO] == scenario]

        # Aggregate MAEs per model
        aggregated = (
            scenario_df.groupby("Model")[["Fact MAE", "Stance MAE"]]
            .mean()
            .round(2)
            .reset_index()
        )

        # Rename columns to include scenario name
        aggregated.rename(
            columns={
                "Fact MAE": f"{scenario}_Fact_MAE",
                "Stance MAE": f"{scenario}_Stance_MAE",
            },
            inplace=True
        )

        # Merge horizontally
        if merged_df is None:
            merged_df = aggregated
        else:
            merged_df = merged_df.merge(
                aggregated,
                on="Model",
                how="outer"
            )
    fact_cols = [f"{scenario}_Fact_MAE" for scenario in SCENARIOS]
    stance_cols = [f"{scenario}_Stance_MAE" for scenario in SCENARIOS]
    merged_df["Average_Fact_MAE"] = (merged_df[fact_cols].mean(axis=1).round(2))
    merged_df["Average_Stance_MAE"] = (merged_df[stance_cols].mean(axis=1).round(2))

    output_path = f"{EVALUATIONS_PATH}/models_mae_summary.csv"
    merged_df.to_csv(output_path, sep=";", index=False)
    print(f"Saved: {output_path}")

def models_percentage():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")
    merged_df = None

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        scenario_df = df[df[SCENARIO] == scenario]

        # Aggregate per model
        aggregated = (
            scenario_df.groupby("Model")[["Fact", "Stance"]]
            .mean()
            .round(2)
            .reset_index()
        )

        # Rename columns to include scenario
        aggregated.rename(
            columns={
                "Fact": f"{scenario}_Fact",
                "Stance": f"{scenario}_Stance",
            },
            inplace=True
        )

        # Merge horizontally across scenarios
        if merged_df is None:
            merged_df = aggregated
        else:
            merged_df = merged_df.merge(
                aggregated,
                on="Model",
                how="outer"
            )

    output_path = f"{EVALUATIONS_PATH}/models_percentage_summary.csv"
    merged_df.to_csv(output_path, sep=";", index=False)
    print(f"Saved: {output_path}")

#Create a table with the MAEs calculated accross language and countries
def language_country_mae():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")
    merged_df = None

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        scenario_df = df[df[SCENARIO] == scenario]

        # ---- Decide grouping depending on scenario ----
        if scenario == SCENARIO_LANGUAGE:
            group_col = LANGUAGES

        elif scenario in SCENARIO_COUNTRY:
            group_col = COUNTRY

        else:
            # fallback: keep full resolution
            scenario_df["Lang_Country"] = scenario_df["languages"].astype(str) + " | " + scenario_df["Country"].astype(str)
            group_col = "Lang_Country"

        # ---- Aggregate (this enforces equality constraint) ----
        aggregated = (
            scenario_df.groupby(group_col)[["Fact MAE", "Stance MAE"]]
            .mean()
            .round(2)
            .reset_index()
        )

        # ---- Rename columns ----
        aggregated.rename(
            columns={
                group_col: "Key",
                "Fact MAE": f"{scenario}_Fact_MAE",
                "Stance MAE": f"{scenario}_Stance_MAE",
            },
            inplace=True
        )

        # ---- Merge across scenarios ----
        if merged_df is None:
            merged_df = aggregated
        else:
            merged_df = merged_df.merge(
                aggregated,
                on="Key",
                how="outer"
            )

    # ---- Final rounding ----
    numeric_cols = merged_df.select_dtypes(include="number").columns
    merged_df[numeric_cols] = merged_df[numeric_cols].round(2)
    
    output_path = f"{EVALUATIONS_PATH}/lang_country_mae_summary.csv"
    merged_df.to_csv(output_path, sep=";", index=False)

    print(f"Saved: {output_path}")
        
#All the results are in the results/evaluations folder
#Creates a table with the model performances in matter of Coherence and Valitidy
model_performances()

#Create a table with the grouped scores of MAE and percentages across model-scenario(lan and country) 
general_stats()

#Create a table with the MAEs calculated accross models
models_mae()

#Create a table with the MAEs calculated accross language and countries
language_country_mae()

#Create a table with the percentage scores accross language and countries
models_percentage()

print(f"✅ Evaluations Done")
