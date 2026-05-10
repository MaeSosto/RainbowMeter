# Create data
from constants import *
import scipy.stats as stats
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

RAINBOW_MAP = "Rainbow Map"
COUNTRY = "Country"
MODEL = "Model"
WEIGHT_LIST = CRITERIA_WEIGHTS_DF["Weight"].values.astype(float).tolist()

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
    
class Evaluations:
    def __init__(self):
        #self.general_stats()
        self.model_performances()
        
    
    def general_stats(self):
        #Iterate on the scenario
        for scenario in SCENARIOS: #SCENARIOS:
            scenario = scenario
            
            percentage_results = {
                MODEL: [],
                COUNTRY: [],
                LANGUAGES: [],
                FACT: [],
                STANCE: [],
                RAINBOW_MAP: [],
            }
            for wil in ["Statistics", "Pvalue", "MAE"]:
                percentage_results[FACT+" "+ wil] = []
                percentage_results[STANCE+" "+ wil] = []
            
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
                        
                        percentage_results[MODEL].append(MODEL_LABEL[model_name])
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
            results_df.to_csv(f"{EVALUATIONS_PATH}/{scenario}/general_stats.csv", sep=";", index=False)

    def model_performances(self):
        # Define metrics once
        metrics = {
            "fact_coh": "Fact Coherence",
            "fact_val": "Fact Validity",
            "fact_coh_val": "Fact Weight coherence by validity",
            "stance_coh": "Stance Coherence",
            "stance_val": "Stance Validity",
            "stance_coh_val": "Stance Weight coherence by validity",
        }
                #Iterate on the scenario
        for scenario in SCENARIOS: #SCENARIOS:
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

                        # compute all means in one go
                        means = existent_rm_df[list(metrics.values())].mean()

                        if scenario == SCENARIO_LANGUAGE:
                            label = language
                        elif scenario == SCENARIO_NATIONALITY:
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

#the results are in the results/evaluations folder
evaluations = Evaluations()