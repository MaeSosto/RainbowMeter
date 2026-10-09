from constants import *
from models import *
import numpy as np

MAX_NUM_ANSWERS = 5 #Num answer we want for each criterion-stance
COHERENCE = "Coherence"
VALIDITY = "Validity"
COH_VAL_SCORE = "Weight coherence by validity" 

class Rainbow_Meter:
    #Return True if the Rainbow map is complete, otherwise return False (and therefore needs to be calculated)
    def __init__(self, model_name):
        self.model_name = model_name
        
        #Iterate on the scenario
        for scenario in SCENARIOS:
            self.scenario = scenario
            
            #Iterate on every country
            for country_name, country_data in COUNTRIES_FILE.items(): # tqdm.tqdm(
                self.country_name = country_name
                self.country_id = country_data[ID]
            
                #Iterate on every language and citizenship 
                for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                    self.language = COUNTRIES_FILE[country_name][LANGUAGES][country_identity_num]
                    self.language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
                    
                    #Retrieve the Rainbow Meter questions of a specific language and scenario
                    rainbow_meter_questions = self.get_raimbow_meter_questions()
                    if rainbow_meter_questions.empty: #If the Rainbow Meter questionnaire in doesn't exist in that language than we cannot compare the results
                        logger.error(f"⚠️ The Rainbow Meter questions file {language} doesn't exist")
                        continue
                    
                    rainbow_meter, num_answers = self.get_rainbow_meter_file_answers()
                    if num_answers == TOT_CRITERIA_NUM:
                        continue
                    
                    #Initialize the model
                    self.model = Model(model_name)
                    error = self.model.initialize_model()
                    if error: #If there are no errors in initializing the model
                        logger.error("Error initializing the model")
                        return None
                    logger.info(f"############ {self.model_name} - {scenario} ############")
                    
                    for idx, row in tqdm.tqdm(rainbow_meter_questions[num_answers:].iterrows(), 
                                            total=TOT_CRITERIA_NUM-num_answers, 
                                            desc=f"🔄 {self.model_name} - {self.scenario} : {f'{language} ({self.language_code})' if self.scenario == SCENARIO_LANGUAGE else f'{self.country_name} ({self.country_id})' if self.scenario == SCENARIO_COUNTRY else f'{self.country_name} in {self.language} ({self.language_code}_{self.country_id})'}",
                                            leave= False
                                            ):
                        rainbow_meter[CRITERION_ID].append(int(idx))
                        
                        for question_type in QUESTION_TYPES:
                            full_prompt, possible_binary_answers = self.get_prompt(row[question_type])
                            # Generate answers
                            question_responses = []
                            while len(question_responses) < MAX_NUM_ANSWERS:
                                resp = self.model.call_model(full_prompt)
                                
                                if resp == None or resp == "":
                                    continue
                                
                                resp_ = self.get_binary_answer(resp, possible_binary_answers)
                                question_responses.append(resp_)

                            rainbow_meter[question_type].append(self.combine_binary_answers(question_responses))

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
        logger.info(f"✅ {self.model_name} answer generations completed!")
    
    #Get the Rainbow Meter file based on the scenario
    def get_raimbow_meter_questions(self):
        result_path = f"{RAINBOW_METER_DATA_PATH}/{self.scenario}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rainbow_meter_{self.language_code}.csv"
        elif self.scenario == SCENARIO_COUNTRY:
            scenario_path = f"rainbow_meter_{self.country_id}.csv"
        else:
            scenario_path = f"rainbow_meter_{self.language_code}_{self.country_id}.csv"
        if os.path.exists(result_path+ scenario_path): #If exist
            df = pd.read_csv(result_path+scenario_path, sep=";")
            #print(f"read: {result_path+ scenario_path}")
            return df
        logger.error(f"⚠️ The Rainbow Meter questions file {result_path+scenario_path} is missing")
        return pd.DataFrame()

    #Return True if the results exists, otherwise False
    def get_rainbow_meter_file_answers(self):
        rainbow_meter = {
            CRITERION_ID: [],
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
        
        result_path = f"{RAINBOW_METER}/{self.scenario}/{self.model_name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rm_answers_{self.language_code}.csv"
        elif self.scenario == SCENARIO_COUNTRY:
            scenario_path = f"rm_answers_{self.country_id}.csv"
        else:
            scenario_path = f"rm_answers_{self.language_code}_{self.country_id}.csv"
        
        if os.path.exists(result_path+scenario_path):
            df = pd.read_csv(result_path+scenario_path, sep=";")#, index_col=CRITERION_ID)
            if df.shape[0] == TOT_CRITERIA_NUM: #Return the pd with all the answers
                logger.info(f"{result_path+scenario_path} complete")
                return rainbow_meter, df.shape[0]
            if df.empty: #There is no file with no answers
                logger.info(f"The Rainbow Meter answers file {result_path+scenario_path} is empty")
                return rainbow_meter, 0
            #The RM exist but it's incomplete, then fill it up until there and continue from there
            for _, row in df.iterrows():
                rainbow_meter[CRITERION_ID].append(int(row[CRITERION_ID]))
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
            return rainbow_meter, df.shape[0]

    #Return True if the results exists, otherwise False
    def rm_result_exist(self):
        result_path = f"{RAINBOW_METER}/{self.scenario}/{self.model_name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rm_answers_{self.language_code}.csv"
        elif self.scenario == SCENARIO_COUNTRY:
            scenario_path = f"rm_answers_{self.country_id}.csv"
        else:
            scenario_path = f"rm_answers_{self.language_code}_{self.country_id}.csv"
        return os.path.exists(result_path+scenario_path), result_path+scenario_path #If a rainbow meter with the looked characteristics exist
    
    #Export and save the Rainbow Meter
    def export_rm_result(self, rainbow_meter):
        result_path = f"{RAINBOW_METER}/{self.scenario}/{self.model_name}/"
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"rm_answers_{self.language_code}.csv"
        elif self.scenario == SCENARIO_COUNTRY:
            scenario_path = f"rm_answers_{self.country_id}.csv"
        else:
            scenario_path = f"rm_answers_{self.language_code}_{self.country_id}.csv"
        os.makedirs(result_path, exist_ok=True)
        rainbow_meter.to_csv(result_path+scenario_path, sep=";", index=False)

    #Return yes/no/unsure/undefined based on the answer
    def get_binary_answer(self, response, answ_options):
        response = response.lower().replace(".", "").replace("*", "").replace('"', '').strip()
        first_word = response.split()[0].strip(",;:!?.")
        if first_word.lower() == answ_options[0].lower():
                return YES
        elif first_word.lower() == answ_options[1].lower():
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
        return round(np.mean(values), 2)
    
            
    def get_prompt(self, criterion):
        with open("data/prompt.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        #The only case where the questions are in english as default
        language = "English" if self.scenario == SCENARIO_COUNTRY else self.language
        lang_data = data[language]
        
        return f"{criterion}\n{lang_data.get('prompt')}", [lang_data.get(YES), lang_data.get(NO)]

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
    

#the results are in the results/rainbow_meter folder
model_list = [QWEN35_2, QWEN35_9, QWEN35_27, LLAMA32_3, DEEPSEEKV32, SONNET46, GPT54, GEMINI3_FLASH]
for model in model_list:
    rainbow_meter = Rainbow_Meter(model)

