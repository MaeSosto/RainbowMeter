from lib.constants import *
from lib.prompt import *

NUM_ATTEMPT = 5 #Num attempt before calling it failed
NUM_ANSWERS = 1 #Num answer we want for each criterion-stance
prompt_types = ["Support", "Opposition", "Fact-Check"]
folder_result = "results_for_analysis/"
folder_language_scenario = "language_scenario/"
path_rainbow_meter = "data/rainbow_meter/"

class Rainbow_Meter:
    def __init__(self, model, country):
        self.country = country
        self.criteria_list = self._get_criteria_list() 
        self.model = model
        logger.info(model.name)
        logger.info(country.name)

    def _get_criteria_list(self):
        criteria_file = path_rainbow_meter+ f'rainbow_meter_{self.country.language}.json'
        criteria_file = open(criteria_file)
        criteria_file = json.load(criteria_file)
        
        #NOW RETURN ONLY COMPLETE CRITERIAS 
        criteria_file = [criteria for criteria in criteria_file if criteria[prompt_types[0]] != ""] 
        return criteria_file
    
    def check_result_already_exist(self):
        path_result = folder_result+folder_language_scenario+self.model.name 
        file_out = f"{path_result}/{self.country.language}_raibow_meter.csv"
        if os.path.exists(file_out):
            file_out = open(file_out)
            file_out = json.load(file_out)
            
            if len(file_out) == len(self.criteria_list):
                return True
        return False

    def export_language_results(self, results):
        json_object = json.dumps(results, indent=4)
        path_result = folder_result+ folder_language_scenario+self.model.name 
        os.makedirs(path_result, exist_ok=True)
        file_out = f"{path_result}/{self.country.language}_raibow_meter.csv"
        with open(file_out, "w") as outfile:
            outfile.write(json_object) 
    
    def get_answers(self):
        results = []
        
        #Iterate on the criteria list
        for criteria in tqdm(self.criteria_list):        
            risp = {
                'Category': criteria['Category'],
                'Subcategory': criteria['Subcategory'],
            }
            logger.info(criteria['Subcategory'])
                
            #Iterate on the prompt types
            for prompt_type in prompt_types:
                self.prompt_settings = Prompt(prompt_type, criteria, self.country)
                risp[prompt_type] = ""
                logger.info(prompt_type)    
                
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
