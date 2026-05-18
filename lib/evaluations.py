# Create data
from constants import *
import scipy.stats as stats
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

WEIGHT_LIST = CRITERIA_WEIGHTS_DF["Weight"].values.astype(float).tolist()

MODELS_PERFORMANCES_PATH = "models_performances"
MAE = "MAE"
PERCENTAGE = "percentage"
EVALUATIONS_PATH = f"evaluations"
os.makedirs(EVALUATIONS_PATH, exist_ok=True)
MAE_PATH =f"{EVALUATIONS_PATH}/{MAE}/"
os.makedirs(MAE_PATH, exist_ok=True)
PERCENTAGE_PATH = f"{EVALUATIONS_PATH}/{PERCENTAGE}/" 
os.makedirs(PERCENTAGE_PATH, exist_ok=True)
for s in SCENARIOS:
    os.makedirs(f"{EVALUATIONS_PATH}/{MODELS_PERFORMANCES_PATH}/{s}", exist_ok=True)
    
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

# Create summary tables for Fact and Stance
def models_mae():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")

    # Preserve model ordering
    ordered_models = [MODEL_LABEL.get(model, model) for model in MODEL_LIST]

    for test in [FACT, STANCE]:
        rows = []
        mae_col = f"{test} {MAE}"
        pvalue_col = f"{test} Pvalue"

        # Iterate through models in fixed order
        for model in MODEL_LIST:
            model_df = df[df["Model"] == model]

            if model_df.empty:
                continue

            row = {"Model": MODEL_LABEL.get(model, model)}

            total_sig = 0
            total_n = 0
            maes = []

            # Aggregate per scenario
            for scenario in SCENARIOS:

                scenario_df = model_df[model_df[SCENARIO] == scenario]

                if scenario_df.empty:
                    row[scenario] = "-"
                    continue

                mae = round(scenario_df[mae_col].mean(), 2)
                sig = (scenario_df[pvalue_col] < 0.05).sum()
                n = len(scenario_df)
                row[scenario] = f"{mae} ({sig}/{n})"

                total_sig += sig
                total_n += n
                maes.append(mae)

            # Average column
            avg_mae = round(np.mean(maes), 2)
            row["Average"] = (f"{avg_mae} ({total_sig}/{total_n})")
            rows.append(row)

        # Create dataframe
        result_df = pd.DataFrame(rows)

        # Preserve ordering
        result_df["Model"] = pd.Categorical(
            result_df["Model"],
            categories=ordered_models,
            ordered=True
        )

        result_df = result_df.sort_values("Model")

        # Reorder columns
        result_df = result_df[["Model"] + SCENARIOS + ["Average"]]

        # Save CSV
        output_path = (f"{EVALUATIONS_PATH}/{MAE}/models_{test}_mae_summary.csv")

        result_df.to_csv(
            output_path,
            sep=";",
            index=False
        )
        print(f"Saved: {output_path}")

#Create a table with the percentage scores accross countries and models
def model_country_percentage():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path, sep=";")

    for test in [FACT, STANCE]:
        # Average stance score for each model-country pair
        stance_table = (
            df.groupby(["Country", "Model"])[test]
            .mean()
            .reset_index()
        )

        # Replace model names with labels
        stance_table["Model"] = stance_table["Model"].map(lambda x: MODEL_LABEL.get(x, x))

        # Pivot models into columns
        pivot = stance_table.pivot(
            index="Country",
            columns="Model",
            values=test
        )

        # Ordered model names
        ordered_models = [MODEL_LABEL.get(model, model) for model in MODEL_LIST]

        # Keep only existing columns and preserve order
        existing_models = [
            model for model in ordered_models
            if model in pivot.columns
        ]

        pivot = pivot[existing_models]

        # Rainbow Map score per country
        rainbow_table = (
            df.groupby("Country")["Rainbow Map"]
            .mean()
            .reset_index()
        )

        # Merge Rainbow Map with model scores
        final_table = rainbow_table.merge(
            pivot,
            on="Country",
            how="left"
        )

        # Sort by country name
        final_table = final_table.sort_values("Country")

        # Remove decimals
        numeric_cols = final_table.select_dtypes(include=["number"]).columns
        final_table[numeric_cols] = (
            final_table[numeric_cols]
            .round(0)
            .astype("Int64")
        )

        # Save output
        output_path = (
            f"{EVALUATIONS_PATH}/percentage/country_model_{test}.csv")
        final_table.to_csv(output_path, sep=";", index=False)

        print(f"Saved: {output_path}")

def models_scenario_percentage():
    csv_path = f"{EVALUATIONS_PATH}/general_stats.csv"

    if not os.path.exists(csv_path):
        print(f"Missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=";")
    merged_df = None

    # Iterate over scenarios contained in the dataframe
    for scenario in SCENARIOS:
        scenario_df = df[df[SCENARIO] == scenario]

        # Replace model names with labels
        scenario_df["Model"] = scenario_df["Model"].map(lambda x: MODEL_LABEL.get(x, x))
        
        # Aggregate per model
        aggregated = (
            scenario_df.groupby("Model")[["Fact", "Stance"]]
            .mean()
            .round(0)
            .astype("Int64")
            .reset_index()
        )
        
        # Ordered model names
        ordered_models = [MODEL_LABEL.get(model, model) for model in MODEL_LIST]

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
        
    # Preserve ordering
    merged_df["Model"] = pd.Categorical(
        merged_df["Model"],
        categories=ordered_models,
        ordered=True
    )

    merged_df = merged_df.sort_values("Model")

    output_path = f"{EVALUATIONS_PATH}/{PERCENTAGE}/models_scenario_percentage.csv"
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
    
    output_path = f"{EVALUATIONS_PATH}/{MAE}/lang_country_mae_summary.csv"
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

#Create a table with the percentage scores accross models and scenarios
models_scenario_percentage()

#Create a table with the percentage scores accross countries and models
model_country_percentage()

print(f"✅ Evaluations Done")
