from lib.constants import *
from lib.prompt import *

NUM_ATTEMPT = 5 #Num attempt before calling it failed
NUM_ANSWERS = 1 #Num answer we want for each criterion-stance
class Rainbow_Meter:
    def __init__(self, model, country):
        self.country = country
        self.criteria_list = _get_criteria_list(country.language) 
        self.model = model
        #logger.info(model.name)
        #logger.info(country.name)

    def export_language_results(self, results):
        json_object = json.dumps(results, indent=4)
        path_result = RESULT_FOLDER+ SCENARIO_LANGUAGE_FOLDER+self.model.name 
        os.makedirs(path_result, exist_ok=True)
        file_out = f"{path_result}/{self.country.language}_raibow_meter.csv"
        with open(file_out, "w") as outfile:
            outfile.write(json_object) 
    
    def get_answers(self):
        results = []
        
        #Iterate on the criteria list
        logger.info(self.model.name +"|"+ self.country.language)
        for criteria in tqdm(self.criteria_list):        
            risp = {
                'Category': criteria['Category'],
                'Subcategory': criteria['Subcategory'],
            }
                
            #Iterate on the prompt types
            for prompt_type in PROMPT_TYPES:
                self.prompt_settings = Prompt(prompt_type, criteria, self.country)
                risp[prompt_type] = ""
                #logger.info(prompt_type)    
                
                #Try with standard prompt    
                attempt = 0
                prompt = self.prompt_settings.get_standard_prompt()
                
                #Try to get a NUM_ANSWERS ok answers until NUM_ATTEMPTS attempts
                while attempt < NUM_ATTEMPT and len(risp[prompt_type]) < NUM_ANSWERS: 
                    response = self.model.call_model(prompt)
                    clean_response = self.prompt_settings.check_response(response)

                    if clean_response != None: #Response is valid
                        risp[prompt_type] = clean_response
                        #logger.info(f"{attempt} | {risp[prompt_type]}")
                        attempt = 0
                        prompt = self.prompt_settings.get_standard_prompt()
                    else:
                        logger.info(f"{attempt} | {response}")
                        #Try with retry prompt
                        attempt += 1
                        prompt = self.prompt_settings.get_retry_prompt(response)
                
                #After 5 attempts return error
                if attempt == NUM_ATTEMPT:
                    logger.error(f"Error: {prompt_type}" )
                    risp[prompt_type] = None
            results.append(risp)
            
            self.export_language_results(results)
            
def _get_criteria_list(language):
    criteria_file = RAINBOW_METER_DATA_PATH+ f'rainbow_meter_{language}.json'
    criteria_file = open(criteria_file)
    criteria_file = json.load(criteria_file)
    
    #NOW RETURN ONLY COMPLETE CRITERIAS 
    criteria_file = [criteria for criteria in criteria_file if criteria[PROMPT_TYPES[0]] != ""] 
    return criteria_file

def check_result_already_exist(model_name, language, ):
    path_result = RESULT_FOLDER+SCENARIO_LANGUAGE_FOLDER+model_name 
    file_out = f"{path_result}/{language}_raibow_meter.csv"
    if os.path.exists(file_out):
        file_out = open(file_out)
        file_out = json.load(file_out)
        
        if len(file_out) == len(_get_criteria_list(language)):
            return True
    return False

##Rainbow Map Queries
def get_country_rainbowmap(country):
    
