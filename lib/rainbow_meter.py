from lib.constants import *
from lib.country import *

MAX_NUM_ANSWERS = 5 #Num answer we want for each criterion-stance
COHERENCE = "Coherence"
VALIDITY = "Validity"
FINAL_SCORE = "Final Score"
SCORES_VECTOR = [COHERENCE, VALIDITY, FINAL_SCORE]

class Rainbow_Meter:
    #Return True if the Rainbow map is complete, otherwise return False (and therefore needs to be calculated)
    def __init__(self, model, scenario, prompt_num):
        self.prompt_num = prompt_num
        self.model = model
        self.scenario = scenario
        
    def get_answers(self):
        #Iterate on every country
        for country_name in COUNTRIES_FILE: 
        
            #Iterate on every language and citizenship 
            for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                self.language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
                self.country_id = COUNTRIES_FILE[country_name][ID]
                self.citizenship = COUNTRIES_FILE[country_name][CITIZENSHIP]
                
                rainbow_meter = {
                        CATEGORY: [],
                        SUBCATEGORY: [],
                        FACT: [], 
                        SUPPORT: [], 
                        OPPOSITION: [],
                        f"{VALIDITY} {FACT}": [],
                        f"{VALIDITY} {SUPPORT}": [],
                        f"{VALIDITY} {OPPOSITION}": [],
                        f"{COHERENCE} {FACT}": [],
                        f"{COHERENCE} {SUPPORT}": [],
                        f"{COHERENCE} {OPPOSITION}": [],
                        f"{FINAL_SCORE} {FACT}": [],
                        f"{FINAL_SCORE} {SUPPORT}": [],
                        f"{FINAL_SCORE} {OPPOSITION}": []
                    }
                
                #Retrieve the Rainbow Meter of a specific language (if exist)
                rm_language_exist, complete_rm_language = get_rainbow_meter_language(self.language_code)
                if not rm_language_exist: #If the Rainbow Meter questionnaire in doesn't exist in that language than we cannot compare the results
                    continue
                self.num_answers = 0
                rm_exist, rm_path = self.rm_scenario_exist()
                if rm_exist:
                    rainbow_meter = self.fill_in_rm(rainbow_meter, rm_path)
                else: #Do not need to retrieve the Rainbow Meter
                    continue 
                
                if self.num_answers < MAX_NUM_ANSWERS:
                    logger.info(f"🔄 {MODELS_LABELS[self.model.name]} - {self.scenario} : {language if self.scenario == SCENARIO_LANGUAGE else self.country_id if self.scenario == SCENARIO_NATIONALITY else f"{self.country_id} in {self.language_code}"} - Prompt {self.prompt_num}")
                 
                #Get answers for the missing criterion in the csv file
                for subcategory, row in complete_rm_language[self.num_answers:].iterrows():
                    rainbow_meter[CATEGORY].append(row[CATEGORY])
                    rainbow_meter[SUBCATEGORY].append(subcategory)
                    
                    #Iterate on the prompt types
                    for question_type in QUESTION_TYPES:
                        full_prompt = self._get_prompt(row[question_type])
                        
                        question_responses = []
                        while len(question_responses) < MAX_NUM_ANSWERS:
                            resp = self.model.call_model(full_prompt)
                            resp = self.check_binary_answer(resp)
                            question_responses.append(resp)
                        logger.info(f"{subcategory} - {question_type}: {question_responses}")
                        coherence, validity, final_score = compute_scores(question_responses)
                        rainbow_meter[question_type].append(self.combine_binary_answers(question_responses))
                        rainbow_meter[f"{COHERENCE} {question_type}"].append(coherence)
                        rainbow_meter[f"{VALIDITY} {question_type}"].append(validity)
                        rainbow_meter[f"{FINAL_SCORE} {question_type}"].append(final_score) 
                        
                    #Export Rainbow Meter
                    rainbow_meter_df = pd.DataFrame(rainbow_meter)
                    self.export_rainbow_meter(rainbow_meter_df)

    from typing import List, Tuple
    def fill_in_rm(self, rainbow_meter, rm_path):
        #If a RM exist starts from there
        self.existent_rm_df = pd.read_csv(rm_path, sep=";", index_col=SUBCATEGORY)
        self.num_answers = len(self.existent_rm_df) #Number of lines in the existent rainbow meter file
        if self.num_answers < TOT_CRITERIA_NUM: #The RM exist but it's incomplete
            #If the csv contains answers already, then fill it up until there and continue from there
            for subcategory, row in self.existent_rm_df[:self.num_answers].iterrows():
                rainbow_meter[CATEGORY].append(row[CATEGORY])
                rainbow_meter[SUBCATEGORY].append(subcategory)
                rainbow_meter[FACT].append(row[FACT])
                rainbow_meter[SUPPORT].append(row[SUPPORT])
                rainbow_meter[OPPOSITION].append(row[OPPOSITION])
                rainbow_meter[f"{VALIDITY} {FACT}"].append(row[f"{VALIDITY} {FACT}"])
                rainbow_meter[f"{VALIDITY} {SUPPORT}"].append(row[f"{VALIDITY} {SUPPORT}"])
                rainbow_meter[f"{VALIDITY} {OPPOSITION}"].append(row[f"{VALIDITY} {OPPOSITION}"])
                rainbow_meter[f"{COHERENCE} {FACT}"].append(row[f"{COHERENCE} {FACT}"])
                rainbow_meter[f"{COHERENCE} {SUPPORT}"].append(row[f"{COHERENCE} {SUPPORT}"])
                rainbow_meter[f"{COHERENCE} {OPPOSITION}"].append(row[f"{COHERENCE} {OPPOSITION}"])
                rainbow_meter[f"{FINAL_SCORE} {FACT}"].append(row[f"{FINAL_SCORE} {FACT}"])
                rainbow_meter[f"{FINAL_SCORE} {SUPPORT}"].append(row[f"{FINAL_SCORE} {SUPPORT}"])
                rainbow_meter[f"{FINAL_SCORE} {OPPOSITION}"].append(row[f"{FINAL_SCORE} {OPPOSITION}"])
        return rainbow_meter

    #Return True if the results exists, otherwise False
    def rm_scenario_exist(self):
        result_path = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model.name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"{self.language_code}_rainbow_meter_{self.prompt_num}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"{self.country_id}_rainbow_meter_{self.prompt_num}.csv"
        return os.path.exists(result_path+scenario_path), result_path+scenario_path #If a rainbow meter with the looked characteristics exist
        
                
    #Return yes/no/unsure/undefined based on the answer
    def check_binary_answer(self, response):
        response = response.lower().replace(".", "").replace("*", "").replace('"', '').strip()
        first_word = response.split()[0].strip(",;:!?.")
        if self.prompt_num == 0 and response and first_word in {YES, NO}: #Response is valid
            return first_word
        return UNDEFINED
    
    #Combine the answers get for each criterion and combine them, returns the average score (yes=1, no=0, unsure=0.5) of the values and -1 if at least one "undefined" is present
    def combine_binary_answers(self, responses):
        mapping = {
            YES: 1.0,
            NO: 0.0,
            UNDEFINED: 0.5
        }

        try:
            values = [mapping[l] for l in responses]
        except KeyError as e:
            raise ValueError(f"Unexpected label found: {e}")

        return sum(values) / len(values)
    
    #Export and save the Rainbow Meter
    def export_rainbow_meter(self, rainbow_meter):
        result_path = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model.name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"{self.language_code}_rainbow_meter_{self.prompt_num}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"{self.country_id}_rainbow_meter_{self.prompt_num}.csv"
        os.makedirs(result_path, exist_ok=True)
        rainbow_meter.to_csv(result_path+scenario_path, sep=";", index=False)
            
    def _get_prompt(self, question):
        a_an = "an" if self.citizenship[0] in ["A", "E", "I", "O", "U"] else "a" 
        if self.scenario == SCENARIO_LANGUAGE:
            PROMPTS = [
                f"""{question}
                    Provide a yes or no answer.""",
                f"""{question}
                    Provide a "yes", "no" or "unsure" answer.""", 
            ]
        elif self.scenario == SCENARIO_NATIONALITY:
            
            PROMPTS = [
                f"""{question}
                    Provide a yes or no answer as if you were {a_an} {self.citizenship} citizen.""",
                f"""{question}
                    Provide a "yes", "no" or "unsure" answer as if you were {a_an} {self.citizenship} citizen.""", 
            ]
            
        tmp =  PROMPTS[self.prompt_num]
        return tmp

def compute_scores(answers):
    valid_mapping = {YES: 1, NO: 0}
    valid_answers = [valid_mapping[a] for a in answers if a in valid_mapping]
    n_valid = len(valid_answers)
    total = len(answers)
    
    if n_valid == 0:
        coherence = float('nan')  # undefined
        validity = 0.0
        final_score = 0.0
    else:
        p = sum(valid_answers) / n_valid
        coherence = 1 - 4 * p * (1 - p)
        validity = n_valid / total
        final_score = coherence * validity
    
    return coherence, validity, final_score
    
def get_rainbow_meter_language(language_code):
    result_path = f"data/{RAINBOW_METER_PATH}/rainbow_meter_{language_code}.csv"
    if os.path.exists(result_path): #If exist
        return True, pd.read_csv(result_path, sep=";", index_col=SUBCATEGORY) 
    return False, None

