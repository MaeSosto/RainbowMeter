# Create data
from lib.constants import *
from lib.country import *
from lib.rainbow_meter import *
import scipy.stats as stats
import numpy as np
from os import listdir
from os.path import isfile, join


class Evaluations:
    def __init__(self, model, scenario, prompt_num):
        self.weights_list = convert_array(CRITERIA_WEIGHTS_DF["Weight"].tolist())
        self.prompt_num = prompt_num
        self.scenario = scenario
        self.model = model
            
    def calculate_wilcoxon(self):
        
        #Scenario Language --> paragono i risultati rainbow meter di scenario language per ogni lingua (e associated country con quella lingua) e i risultati della rainbow map di quello stesso country  
        #Scenario Nationality --> paragono i risultati rainbow meter di scenario nationality per ogni country e i risultati della rainbow map di quello stesso country
        #Scenario Nationality --> paragono i risultati rainbow meter di scenario language + nationality per ogni country e i risultati della rainbow map di quello stesso country
    

        #Get the languages of the rainbow meters retreived in the language scenario  
        path_rainbow_meters = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model.name}/" 
        rm_lanaguages = list(set([f.split("_")[0] for f in listdir(path_rainbow_meters) if isfile(join(path_rainbow_meters, f))]))


        #Get the array score of every country, considering only the language now, and compare it to the scores obtained on the rainbow map
        results = {
            "Country": [],
            "country_id": [],
            "statistics": [], 
            "pvalue": [] 
            }
        
        # #country_list = [Country(c) for c in COUNTRIES_FILE if "en" in COUNTRIES_FILE[c][LANGUAGES_CODE]]
        # country_list = get_countries_that_speaks()
        
        for country in COUNTRIES_FILE:
            print(country.name)
            rainbow_map_country = convert_array(RAINBOW_MAP_DF.loc[country.id].drop("country_name").drop("Rank").values)
            rainbow_map_country = np.multiply(rainbow_map_country, self.weights_list)
            #print(rainbow_map_country)
            #criteria_file = country.criteria_file
            
            exist, rainbow_meter = self._get_rainbow_meter_results(country)
            if not exist:
                continue
            
            rainbow_meter_country = rainbow_meter[FACT].values
            rainbow_meter_country = np.multiply(rainbow_meter_country, self.weights_list)
            #print(rainbow_meter_country)
            
            wilcoxon_s, wilcoxon_pval = self.wilcoxon(rainbow_map_country, rainbow_meter_country)
            
            results["Country"].append(country.name)
            results["country_id"].append(country.id)
            results["statistics"].append(wilcoxon_s)
            results["pvalue"].append(wilcoxon_pval)
            #results.append(row)
            print(f"{wilcoxon_s}, {wilcoxon_pval}")
                
            
        results = pd.DataFrame(results)
        print(results)
        self.export_csv(country, results)
        
        
    def export_csv(self, country, df):
        result_path = f"{RESULT_PATH}/{EVALUATIONS_PATH}/{self.scenario}/{self.model.name}/"
        os.makedirs(result_path, exist_ok=True)
        df.to_csv(f"{result_path}{country.language}_wilcoxon_{self.prompt_num}.csv", sep=";", index=False)

        
    # conduct the Wilcoxon-Signed Rank Test
    def wilcoxon(group1, group2):
        return stats.wilcoxon(group1, group2).statistic, stats.wilcoxon(group1, group2).pvalue

    def _get_rainbow_meter_results(self, country):
        result_path = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model_name}/{country.language}_rainbow_meter_{self.prompt_num}.csv"
        if os.path.exists(result_path):
            pd_rm = pd.read_csv(result_path, delimiter=";", index_col="Subcategory")
            num_rows = len(pd_rm)
            if num_rows == TOT_CRITERIA_NUM:  
                return True, pd_rm
        return False, 0

def convert_array(arr):
    # Convert everything to float
    return np.array([
        float(str(x).replace(",", "."))
        for x in arr
    ])

