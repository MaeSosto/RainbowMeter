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

        self.plain_result()
    
    def plain_result(self):

        results = {
            "Country": [],
            "Predicted": [],
            "Real": [],
            "Error": []
        }
        for cat in RAINBOW_MAP_CATEGORIES:
            results[cat] = []
        
        for country_name in COUNTRIES_FILE: 
        
            #Iterate on every language and citizenship 
            for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                self.language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
                self.country_id = COUNTRIES_FILE[country_name][ID]
                self.citizenship = COUNTRIES_FILE[country_name][CITIZENSHIP]
                
                
                #Retrieve the Rainbow Meter of a specific language (if exist)
                rm_exist, rm_path = self.rm_scenario_exist()
                if rm_exist:
                    #RM exist
                    self.existent_rm_df = pd.read_csv(rm_path, sep=";", index_col=SUBCATEGORY)
                    self.num_answers = len(self.existent_rm_df) #Number of lines in the existent rainbow meter file
                    if self.num_answers < TOT_CRITERIA_NUM: #RM is incomplete
                        continue
                else: #RM doesn't exist
                    continue
                                
                country_real_scores = []
                country_predicted_scores = []
                for cat in RAINBOW_MAP_CATEGORIES:
                    self.cat = cat
                    self.subcat = [subcategory for subcategory, row in CRITERIA_WEIGHTS_DF[CRITERIA_WEIGHTS_DF[CATEGORY] == cat].iterrows()]
                    self.weights_list = [CRITERIA_WEIGHTS_DF.loc[sub]["Weight"] for sub in self.subcat]
                    
                    tmp = self.get_rainbow_map_category_score().astype(float)
                    tmp_ = sum(tmp) 
                    country_real_scores = np.append(country_real_scores, tmp)
                    results_cat = self.get_rainbow_meter_category_score()
                    country_predicted_scores = np.append(country_predicted_scores, results_cat)
                    results[cat].append(float(sum(results_cat)))
                
                results["Country"].append(country_name)
                results["Real"].append(float(sum(country_real_scores)))
                results["Predicted"].append(float(sum(country_predicted_scores)))
                results["Error"].append(float(np.linalg.norm(country_real_scores - country_predicted_scores, 1)))    
                
                #Export Results
                results_df = pd.DataFrame(results)
                self.export_plain_result(results_df)

    #Export and save plain results            
    def export_plain_result(self, plain_results):
        result_path = f"{RESULT_PATH}/{EVALUATIONS_PATH}/{self.scenario}/{self.model.name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"{self.language_code}_plain_results_{self.prompt_num}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"{self.country_id}_plain_results_{self.prompt_num}.csv"
        os.makedirs(result_path, exist_ok=True)
        plain_results.to_csv(result_path+scenario_path, sep=";", index=False)
                
    
    def get_rainbow_map_category_score(self):
        rainbow_map_country = RAINBOW_MAP_DF.loc[self.country_id].drop("country_name").drop("Rank")
        rainbow_map_scores = [rainbow_map_country[sub] for sub in self.subcat]
        weights_list = self.weights_list
        if self.cat == "Family":
            marriage_scores = rainbow_map_scores[:4]
            max_num = -1
            weight_max = -1
            for idx, sco in enumerate(reversed(marriage_scores)):
                if sco > max_num:
                    max_num = np.int64(sco)
                    weight_max = np.float64(self.weights_list[3-idx])
            rainbow_map_scores = np.insert(rainbow_map_scores[4:], 0, max_num)
            weights_list = np.insert(self.weights_list[4:], 0, weight_max)
        return np.multiply(rainbow_map_scores, weights_list)

    def get_rainbow_meter_category_score(self):
        existent_rm = [self.existent_rm_df.loc[sub][FACT] for sub in self.subcat]
        weights_list = self.weights_list
        if self.cat == "Family":
            marriage_scores = existent_rm[:4]
            max_num = -1
            weight_max = -1
            for idx, sco in enumerate(reversed(marriage_scores)):
                if sco >= max_num:
                    max_num = np.int64(sco)
                    weight_max = np.float64(self.weights_list[3-idx])
            existent_rm = np.insert(existent_rm[4:], 0, max_num)
            weights_list = np.insert(self.weights_list[4:], 0, weight_max)
        return np.multiply(existent_rm, weights_list)
    
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
    
    def rm_scenario_exist(self):
        result_path = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model.name}/"
        
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"{self.language_code}_rainbow_meter_{self.prompt_num}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"{self.country_id}_rainbow_meter_{self.prompt_num}.csv"
        return os.path.exists(result_path+scenario_path), result_path+scenario_path #If a rainbow meter with the looked characteristics exist    
        
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
    return [
        float(str(x).replace(",", "."))
        for x in arr
    ]

