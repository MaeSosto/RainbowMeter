# Create data
from lib.constants import *
from lib.country import *
from lib.rainbow_meter import *
import scipy.stats as stats
import numpy as np

def get_evaluations(model_name, prompt_num, country_list):
    weights_list = convert_array(CRITERIA_WEIGHTS_DF["Weight"].tolist())
    
    #results = []
    #Get the array score of every country, considering only the language now, and compare it to the scores obtained on the rainbow map
    row = {
        "Country": [],
        "country_id": [],
        "statistics": [], 
        "pvalue": [] 
        }
    for country in country_list:
        
        print(country.name)
        rainbow_map_country = convert_array(RAINBOW_MAP_DF.loc[country.id].drop("country_name").drop("Rank").values)
        rainbow_map_country = np.multiply(rainbow_map_country, weights_list)
        #print(rainbow_map_country)
        #criteria_file = country.criteria_file
        
        exist, rainbow_meter = get_rainbow_meter_results(model_name, country.language, prompt_num)
        if exist:
            rainbow_meter_country = rainbow_meter[QUESTION_FACT].values
            rainbow_meter_country = np.multiply(rainbow_meter_country, weights_list)
            #print(rainbow_meter_country)
            
            wilcoxon_s, wilcoxon_pval = wilcoxon(rainbow_map_country, rainbow_meter_country)
            row["Country"].append(country.name)
            row["country_id"].append(country.id)
            row["statistics"].append(wilcoxon_s)
            row["pvalue"].append(wilcoxon_pval)
            #results.append(row)
            print(f"{wilcoxon_s}, {wilcoxon_pval}")
            
        else:
            continue
    results = pd.DataFrame(row)
    print(results)
    export_csv(model_name, country, prompt_num, results)
    
    
def export_csv(model_name, country, prompt_num, df):
    result_path = f"{RESULT_PATH}/{EVALUATIONS_PATH}/{model_name}/"
    os.makedirs(result_path, exist_ok=True)
    df.to_csv(f"{result_path}{country.language}_wilcoxon_{prompt_num}.csv", sep=";", index=False)

    
# conduct the Wilcoxon-Signed Rank Test
def wilcoxon(group1, group2):
    return stats.wilcoxon(group1, group2).statistic, stats.wilcoxon(group1, group2).pvalue

def convert_array(arr):
    # Convert everything to float
    return np.array([
        float(str(x).replace(",", "."))
        for x in arr
    ])

def get_rainbow_meter_results(model_name, language, prompt_num):
    result_path = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{SCENARIO_LANGUAGE_PATH}/{model_name}/{language}_rainbow_meter_{prompt_num}.csv"
    if os.path.exists(result_path):
        pd_rm = pd.read_csv(result_path, delimiter=";", index_col="Subcategory")
        num_rows = len(pd_rm)
        if num_rows == CRITERIA_NUM:  
            return True, pd_rm
    return False, 0
