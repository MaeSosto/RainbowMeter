from lib.constants import *
from lib.prompt import *
from lib.models import *

NUM_ATTEMPT = 5 #Num attempt before calling it failed
NUM_ANSWERS = 1 #Num answer we want for each criterion-stance
CLASSIFIER_MODEL = LLAMA3 #Model used to classifier the answers in a binary format

class Rainbow_Meter:
    def __init__(self, model_name, country):
        self.country = country
        self.criteria_list = _get_criteria_list(country.language) 
        self.model_name = model_name
        #logger.info(model.name)
        #logger.info(country.name)

    def export_language_results(self, results):
        json_object = json.dumps(results, indent=4)
        path_result = RESULT_FOLDER+ SCENARIO_LANGUAGE_FOLDER+self.model_name 
        os.makedirs(path_result, exist_ok=True)
        file_out = f"{path_result}/{self.country.language}_raibow_meter.json"
        with open(file_out, "w") as outfile:
            outfile.write(json_object) 
    
    def get_answers(self):
        results = []
        classifier = Model(CLASSIFIER_MODEL)
        classifier.initialize_model()
        
        self.model = Model(self.model_name)
        error = self.model.initialize_model()
        
        if error: #If there are no errors in initializing the model
            return True  
        
        #Iterate on the criteria list
        #logger.info(self.model_name +"|"+ self.country.language)
        for criteria in tqdm(self.criteria_list):        
            risp = {
                'Category': criteria['Category'],
                'Subcategory': criteria['Subcategory'],
            }
                
            #Iterate on the prompt types
            for prompt_type in PROMPT_TYPES:
                self.prompt_settings = Prompt(prompt_type, criteria, self.country)
                #logger.info(prompt_type)    
                
                #Try with standard prompt    
                attempt = 0
                prompt = criteria[prompt_type] #self.prompt_settings.get_standard_prompt()
                risp[prompt_type] = []
                risp[prompt_type+" attempt"] = attempt+1
                risp[prompt_type+" binary"] = []
                
                #Reprompt until 5 times
                while attempt < NUM_ATTEMPT: 
                    response = self.model.call_model(prompt)
                    #clean_response = self.prompt_settings.check_response(response)
                    binary_response = classifier.get_binary_answer(response)

                    risp[prompt_type].append(response) 
                    risp[prompt_type+" attempt"] = attempt+1
                    risp[prompt_type+" binary"].append(binary_response)
                    if (response != None and response != "") and (binary_response == "yes" or binary_response == "no"): #Response is valid
                        break
                    else: #Response is not valid
                        #logger.info(f"{attempt+1} | {criteria["Subcategory"]}[{prompt_type}]")
                        #Try with retry prompt
                        attempt += 1
                        prompt = self.prompt_settings.retry_prompt(response)
                        
                #After 5 attempts return error
                if attempt == NUM_ATTEMPT:
                    logger.error(f"Error: {prompt_type}" )
                    risp[prompt_type+" attempt"] = attempt+1
                    risp[prompt_type].append(response)
                    risp[prompt_type+" binary"].append(binary_response)
            results.append(risp)
            
            self.export_language_results(results)
        return False
    
def process_binary_answers(model_name, country):
    classifier = Model(CLASSIFIER_MODEL)
    classifier.initialize_model()
    path_result = RESULT_FOLDER+ SCENARIO_LANGUAGE_FOLDER+model_name
    file = f"{path_result}/{country.language}_raibow_meter.json"
    file = open(file)
    file = json.load(file)
    for criterion_item in file:
        for prompt_type in PROMPT_TYPES:
            response = criterion_item[prompt_type]
            response = classifier.get_binary_answer(response)
            criterion_item[prompt_type+ " binary"] = response
        export_language_results(file, model_name, country)

def export_language_results(results, model_name, country):
        json_object = json.dumps(results, indent=4)
        path_result = RESULT_FOLDER+ SCENARIO_LANGUAGE_FOLDER+model_name 
        os.makedirs(path_result, exist_ok=True)
        file_out = f"{path_result}/{country.language}_raibow_meter.json"
        with open(file_out, "w") as outfile:
            outfile.write(json_object) 
            
def _get_criteria_list(language):
    criteria_file = RAINBOW_METER_DATA_PATH+ f'rainbow_meter_{language}.json'
    criteria_file = open(criteria_file)
    criteria_file = json.load(criteria_file)
    
    #NOW RETURN ONLY COMPLETE CRITERIAS 
    criteria_file = [criteria for criteria in criteria_file if criteria[PROMPT_TYPES[0]] != "" and criteria[PROMPT_TYPES[0]] != None] 
    return criteria_file

#Return True if the file already axist and it has the same number of criterion as the the one in the original of criteria, else False
def check_result_already_exist(model_name, language):
    path_result = RESULT_FOLDER+SCENARIO_LANGUAGE_FOLDER+model_name 
    file_out = f"{path_result}/{language}_raibow_meter.json"
    if os.path.exists(file_out):
        try:
            file_out = open(file_out)
            file_out = json.load(file_out)
        
            if len(file_out) == len(_get_criteria_list(language)):
                return True
        except Exception as X:
            return False
    return False

#Return True if all the criterion has the binary answers, else False
def check_binary_answers(model_name, language):
    file_out = f"{RESULT_FOLDER+SCENARIO_LANGUAGE_FOLDER+model_name}/{language}_raibow_meter.json"
    file_out = open(file_out)
    file_out = json.load(file_out)
    try:
        ok_answers = [
            answ
            for answ in file_out
            if any(answ[prompt+" binary"] not in ["", None] for prompt in PROMPT_TYPES)
        ]
        if len(ok_answers) == len(_get_criteria_list(language)):
            return True
        return False
    except Exception as X:
        return False