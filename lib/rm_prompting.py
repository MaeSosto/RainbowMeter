from constants import *
from models import *
import numpy as np

MAX_NUM_ANSWERS = 5 #Num answer we want for each criterion-stance
COHERENCE = "Coherence"
VALIDITY = "Validity"
COH_VAL_SCORE = "Weight coherence by validity"

class Rainbow_Meter:
    #Return True if the Rainbow map is complete, otherwise return False (and therefore needs to be calculated)
    def __init__(self, model, scenario):
        self.model = model
        self.scenario = scenario
    
        #Return True if the results exists, otherwise False
    def rm_result_exist(self):
        result_path = f"{RAINBOW_METER_RESULT_PATH}/{self.scenario}/{self.model.model_name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rm_answers_{self.language_code}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"rm_answers_{self.country_id}.csv"
        else:
            scenario_path = f"rm_answers_{self.language_code}_{self.country_id}.csv"
        return os.path.exists(result_path+scenario_path), result_path+scenario_path #If a rainbow meter with the looked characteristics exist
        
    #Export and save the Rainbow Meter
    def export_rm_result(self, rainbow_meter):
        result_path = f"{RAINBOW_METER_RESULT_PATH}/{self.scenario}/{self.model.model_name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rm_answers_{self.language_code}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"rm_answers_{self.country_id}.csv"
        else:
            scenario_path = f"rm_answers_{self.language_code}_{self.country_id}.csv"
        os.makedirs(result_path, exist_ok=True)
        rainbow_meter.to_csv(result_path+scenario_path, sep=";", index=False)
    
    #Get the Rainbow Meter file based on the scenario
    def get_rainbow_meter(self):
        result_path = f"{RAINBOW_METER_DATA_PATH}/{self.scenario}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rainbow_meter_{self.language_code}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"rainbow_meter_{self.country_id}.csv"
        else:
            scenario_path = f"rainbow_meter_{self.language_code}_{self.country_id}.csv"
        if os.path.exists(result_path+ scenario_path): #If exist
            df = pd.read_csv(result_path+scenario_path, sep=";", index_col=SUBCATEGORY)
            return True,  df
        return False, None
    
    def get_answers(self):
        #Iterate on every country
        for country_name, country_data in tqdm.tqdm(COUNTRIES_FILE.items(), total=len(COUNTRIES_FILE), desc=f"🔄 {self.model.model_name} - {self.scenario}"):
            self.country_name = country_name
            self.country_id = country_data[ID]
            self.citizenship = country_data[CITIZENSHIP]
        
            #Iterate on every language and citizenship 
            for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                self.language = COUNTRIES_FILE[country_name][LANGUAGES][country_identity_num]
                self.language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
                
                rainbow_meter = {
                        CATEGORY: [],
                        SUBCATEGORY: [],
                        FACT: [], 
                        SUPPORT: [], 
                        OPPOSITION: [],
                        f"{STANCE}" : [],
                        f"{FACT} {COHERENCE}" : [],
                        f"{FACT} {VALIDITY}" : [],
                        f"{FACT} {COH_VAL_SCORE}" : [],
                        f"{STANCE} {COHERENCE}" : [],
                        f"{STANCE} {VALIDITY}" : [],
                        f"{STANCE} {COH_VAL_SCORE}"  : [],
                    }
                
                #Retrieve the Rainbow Meter of a specific language (if exist)
                #if self.scenario == SCENARIO_LANGUAGE:
                rm_language_exist, complete_rm_language = self.get_rainbow_meter()
                if not rm_language_exist: #If the Rainbow Meter questionnaire in doesn't exist in that language than we cannot compare the results
                    continue
                # elif self.scenario == SCENARIO_NATIONALITY:
                #     rm_language_exist, complete_rm_language = self.get_rainbow_meter()
                
                self.num_answers = 0
                rm_exist, rm_path = self.rm_result_exist()
                if rm_exist:
                    rainbow_meter = self.fill_in_rm(rainbow_meter, rm_path)
                
                #if self.num_answers < MAX_NUM_ANSWERS:
                    #logger.info(f"🔄 {self.model.model_name} - {self.scenario} : {language if self.scenario == SCENARIO_LANGUAGE else self.country_id if self.scenario == SCENARIO_NATIONALITY else f"{self.country_id} in {self.language_code}"}")
                
                #Get answers for the missing criterion in the csv file
                for subcategory, row in tqdm.tqdm(complete_rm_language[self.num_answers:].iterrows(), 
                                                total=len(complete_rm_language[self.num_answers:]), 
                                                desc=f"🔄 {self.model.model_name} - {self.scenario} : {language if self.scenario == SCENARIO_LANGUAGE else self.country_id if self.scenario == SCENARIO_NATIONALITY else f"{self.country_id} in {self.language_code}"}",
                                                leave= False
                    ):
                    rainbow_meter[CATEGORY].append(row[CATEGORY])
                    rainbow_meter[SUBCATEGORY].append(subcategory)
                    
                    for question_type in QUESTION_TYPES:
                        full_prompt, possible_binary_answers = self.get_prompt(row[question_type])

                        # Generate answers
                        question_responses = []
                        while len(question_responses) < MAX_NUM_ANSWERS:
                            resp = self.model.call_model(full_prompt)
                            if resp == None or resp == "":
                                continue
                            resp = self.get_binary_answer(resp, possible_binary_answers)
                            question_responses.append(resp)

                        rainbow_meter[question_type].append(round(self.combine_binary_answers(question_responses),2))

                        if question_type == OPPOSITION:
                            rainbow_meter[f"{STANCE}"].append(round(np.mean([rainbow_meter[SUPPORT][-1], np.abs(rainbow_meter[OPPOSITION][-1] - 1)]), 2))
                            
                        if question_type in {FACT, OPPOSITION}:
                            coherence, validity, final_score = model_scores(question_responses)
                            prefix = FACT if question_type == FACT else STANCE
                            rainbow_meter[f"{prefix} {COHERENCE}"].append(round(coherence, 2))
                            rainbow_meter[f"{prefix} {VALIDITY}"].append(round(validity, 2))
                            rainbow_meter[f"{prefix} {COH_VAL_SCORE}"].append(round(final_score, 2))
                            
                    #Export Rainbow Meter
                    rainbow_meter_df = pd.DataFrame(rainbow_meter)
                    self.export_rm_result(rainbow_meter_df)

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
                rainbow_meter[f"{STANCE}"].append(row[f"{STANCE}"])
                rainbow_meter[f"{FACT} {COHERENCE}"].append(row[f"{FACT} {COHERENCE}"])
                rainbow_meter[f"{FACT} {VALIDITY}"].append(row[f"{FACT} {VALIDITY}"])
                rainbow_meter[f"{FACT} {COH_VAL_SCORE}"].append(row[f"{FACT} {COH_VAL_SCORE}"])
                rainbow_meter[f"{STANCE} {COHERENCE}"].append(row[f"{STANCE} {COHERENCE}"])
                rainbow_meter[f"{STANCE} {VALIDITY}"].append(row[f"{STANCE} {VALIDITY}"])
                rainbow_meter[f"{STANCE} {COH_VAL_SCORE}"].append(row[f"{STANCE} {COH_VAL_SCORE}"])
        return rainbow_meter


                
    #Return yes/no/unsure/undefined based on the answer
    def get_binary_answer(self, response, answ_options):
        response = response.lower().replace(".", "").replace("*", "").replace('"', '').strip()
        first_word = response.split()[0].strip(",;:!?.")
        if response and first_word in answ_options: #Response is valid
            if first_word == answ_options[0]:
                return YES
            return NO
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

        return np.mean(values)
    
            
    def get_prompt(self, criterion):
        with open("data/prompt.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        #The only case where the questions are in english as default
        language = "English" if self.scenario == SCENARIO_NATIONALITY else self.language
        lang_data = data[language]
        
        return f"{criterion}\n{lang_data.get("prompt")}", [lang_data.get(YES), lang_data.get(NO)]

def model_scores(answers):
    valid_mapping = {YES: 1, NO: 0}
    valid_answers = [valid_mapping[a] for a in answers if a in valid_mapping]
    n_valid = len(valid_answers)
    
    if n_valid == 0:
        coherence = float('nan')  # undefined
        validity = 0.0
        final_score = 0.0
    else:
        p = sum(valid_answers) / n_valid
        coherence = 1 - 4 * p * (1 - p)
        validity = n_valid / MAX_NUM_ANSWERS
        final_score = coherence * validity
    
    return coherence, validity, final_score
    





model_list = [SONNET46]

#Iterate on Models
for model_name in model_list: #tqdm.tqdm(model_list, desc="Answering Rainbow Meter Criteria", total=len(model_list)):
    model = Model(model_name)
    error = model.initialize_model()
    if error: #If there are no errors in initializing the model
        logger.info("Error initializing the model")
        break
    
    #Iterate on the scenario
    for scenario in SCENARIOS:
        rainbow_meter = Rainbow_Meter(model, scenario)
        rainbow_meter.get_answers()
            
            # #Evaluations
            # logger.info("🧮 Evaluations")
            # eval = Evaluations(model, scenario, PROMPT_NUM)
            # # eval.calculate_wilcoxon()
        logger.info(f"✅ {scenario} Done")    

if not error:
    logger.info("✅ All models done")