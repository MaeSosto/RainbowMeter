# Create data
from constants import *
from models import *
import scipy.stats as stats
import numpy as np
from os import listdir
from os.path import isfile, join

RAINBOW_MAP = "Rainbow Map"
COUNTRY = "Country"

def convert_array(arr):
    return [
        float(str(x).replace(",", "."))
        for x in arr
    ]
    
class Evaluations:
    def __init__(self, model_name):
        self.weights_list = CRITERIA_WEIGHTS_DF["Weight"].values
        
        self.model = Model(model_name)
        error = self.model.initialize_model()
        if error: #If there are no errors in initializing the model
            logger.info("Error initializing the model")
            return None
        
        #Iterate on the scenario
        for scenario in [SCENARIO_LANGUAGE]: #SCENARIOS:
            self.scenario = scenario
                
            results = {
                COUNTRY: [],
                FACT: [],
                STANCE: [],
                RAINBOW_MAP: [],
                #"Error": []
            }
            
            #Iterate on every country
            for country_name, country_data in tqdm.tqdm(
                    COUNTRIES_FILE.items(),
                    total=len(COUNTRIES_FILE),
                    desc=f"🔄 {self.model.model_name} - {self.scenario}"
                ):
                self.country_name = country_name
                self.country_id = country_data[ID]
                self.citizenship = country_data[CITIZENSHIP]
            
                #Iterate on every language and citizenship 
                for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                    self.language = COUNTRIES_FILE[country_name][LANGUAGES][country_identity_num]
                    self.language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
                
                    #Retrieve the Rainbow Meter of a specific language (if exist)
                    self.existent_rm_df = self.get_rainbow_map()
                    if self.existent_rm_df.empty:  #RM doesn't exist or is incomplete
                        return None
                    
                    results[COUNTRY].append(country_name)
                    
                    #Get the Rainbow Map real Scores of that country
                    rainbow_map_scores = RAINBOW_MAP_DF.loc[self.country_id].drop("country_name").drop("Rank").values.astype(float).tolist()
                    #Get the rainbow Meter scores of that country
                    rainbow_meter_fact_scores = self.existent_rm_df[FACT].values.astype(float).tolist()
                    rainbow_meter_supp_scores = self.existent_rm_df[SUPPORT].values.astype(float).tolist()
                    rainbow_meter_opp_scores = self.existent_rm_df[OPPOSITION].values.astype(float).tolist()
                    
                    rainbow_map_scores, w1 = self.family_workaround(rainbow_map_scores)
                    rainbow_meter_fact_scores, w2 = self.family_workaround(rainbow_meter_fact_scores)
                    
                    results[RAINBOW_MAP].append(float(sum(np.multiply(rainbow_map_scores, w1))))
                    results[FACT].append(float(sum(np.multiply(rainbow_meter_fact_scores, w2))))
                    #results["Error"].append(float(np.linalg.norm(rainbow_map_scores - rainbow_meter_scores, 1)))    
                        
                    #Export Results
                    results_df = pd.DataFrame(results)
                    self.export_plain_result(results_df)
    
    #Return True if the results exists, otherwise False
    def get_rainbow_map(self):
        result_path = f"{RAINBOW_METER_RESULT_PATH}/{self.scenario}/{self.model.model_name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rm_answers_{self.language_code}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"rm_answers_{self.country_id}.csv"
        else:
            scenario_path = f"rm_answers_{self.language_code}_{self.country_id}.csv"
        if os.path.exists(result_path+scenario_path):
            df = pd.read_csv(result_path+scenario_path, sep=";", index_col=SUBCATEGORY) 
            if df.shape[0] == TOT_CRITERIA_NUM:
                return df
            else:
                logger.error(f"⚠️ {result_path+scenario_path} is incomplete")
        else:
            logger.error(f"⚠️ {result_path+scenario_path} is missing")
        return pd.DataFrame()

    
    #Get the full list of scores and adjust them accordingly to the Family category contraints
    def family_workaround(self, rm_scores):
        #Workaround for the marriage criteria
        family_weight_list = self.weights_list[25:29]
        family_rm_scores = rm_scores[25:29]
        max_num = 0
        for idx, sco in enumerate(reversed(family_rm_scores)):
            if sco >= max_num:
                max_num = sco
                weight_max = family_weight_list[3-idx]
        
        tmp1 = np.insert(rm_scores[29:], 0, max_num)
        tmp2 = np.insert(self.weights_list[29:], 0, weight_max)
        return np.concatenate([rm_scores[:25], tmp1]), np.concatenate([self.weights_list[:25], tmp2])
                
    #Export and save plain results            
    def export_plain_result(self, plain_results):
        result_path = f"{RESULT_PATH}/{EVALUATIONS_PATH}/{self.scenario}/{self.model.model_name}/"
        os.makedirs(result_path, exist_ok=True)
        
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"plain_results_{self.language_code}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"plain_results_{self.language_code}.csv"
        else:
            scenario_path = f"plain_results_{self.language_code}_{self.country_id}.csv"
            
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

    def calculate_wilcoxon(self):
        
        #Scenario Language --> paragono i risultati rainbow meter di scenario language per ogni lingua (e associated country con quella lingua) e i risultati della rainbow map di quello stesso country  
        #Scenario Nationality --> paragono i risultati rainbow meter di scenario nationality per ogni country e i risultati della rainbow map di quello stesso country
        #Scenario Nationality --> paragono i risultati rainbow meter di scenario language + nationality per ogni country e i risultati della rainbow map di quello stesso country
    

        #Get the languages of the rainbow meters retreived in the language scenario  
        path_rainbow_meters = f"{RAINBOW_METER_RESULT_PATH}/{self.scenario}/{self.model.model_name}/" 
        rm_lanaguages = list(set([f.split("_")[0] for f in listdir(path_rainbow_meters) if isfile(join(path_rainbow_meters, f))]))


        #Get the array score of every country, considering only the language now, and compare it to the scores obtained on the rainbow map
        results = {
            "Country": [],
            "country_id": [],
            "statistics": [], 
            "pvalue": [] 
            }
        
        
        for country in COUNTRIES_FILE:
            print(country.name)
            rainbow_map_country = convert_array(RAINBOW_MAP_DF.loc[country.id].drop("country_name").drop("Rank").values)
            rainbow_map_country = np.multiply(rainbow_map_country, self.weights_list)
            
            exist, rainbow_meter = self._get_rainbow_meter_results(country)
            if not exist:
                continue
            
            rainbow_meter_country = rainbow_meter[FACT].values
            rainbow_meter_country = np.multiply(rainbow_meter_country, self.weights_list)
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
        result_path = f"{RESULT_PATH}/{EVALUATIONS_PATH}/{self.scenario}/{self.model.model_name}/"
        os.makedirs(result_path, exist_ok=True)
        df.to_csv(f"{result_path}{country.language}_wilcoxon.csv", sep=";", index=False)

        
    # conduct the Wilcoxon-Signed Rank Test
    def wilcoxon(group1, group2):
        return stats.wilcoxon(group1, group2).statistic, stats.wilcoxon(group1, group2).pvalue

    def _get_rainbow_meter_results(self, country):
        result_path = f"{RAINBOW_METER_RESULT_PATH}/{self.scenario}/{self.model_name}/{country.language}_rainbow_meter.csv"
        if os.path.exists(result_path):
            pd_rm = pd.read_csv(result_path, delimiter=";", index_col="Subcategory")
            if pd_rm.shape[0] == TOT_CRITERIA_NUM:  
                return True, pd_rm
        return False, 0
    
    
model_name = DEEPSEEKV32
evaluations = Evaluations(model_name)

