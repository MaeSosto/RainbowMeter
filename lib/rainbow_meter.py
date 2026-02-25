from lib.constants import *
from lib.country import *
from lib.evaluations import *

MAX_NUM_ANSWERS = 5 #Num answer we want for each criterion-stance
COHERENCE = "Coherence"
VALIDITY = "Validity"

class Rainbow_Meter:
    #Return True if the Rainbow map is complete, otherwise return False (and therefore needs to be calculated)
    def __init__(self, model, country, language, scenario, prompt_num):
        self.prompt_num = prompt_num
        self.model = model
        self.country = country
        self.language = language
        self.scenario = scenario
        
    def get_answers(self):
        logger.info(f"🔄 {MODELS_LABELS[self.model.name]} - {self.country.id} - {self.language}")
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
            f"{COHERENCE} {OPPOSITION}": []
        }
        
        #If the csv contains answers already, then fill it up until there and continue from there
        for subcategory, row in self.rainbow_meter_questions[:self.num_criteria_filled].iterrows():
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
            
        
        #Get answers for the missing criterion in the csv file
        for subcategory, row in self.rainbow_meter_questions[self.num_criteria_filled:].iterrows():
            rainbow_meter[CATEGORY].append(row[CATEGORY])
            rainbow_meter[SUBCATEGORY].append(subcategory)
            
            #Iterate on the prompt types
            for question_type in QUESTION_TYPES:
                full_prompt = self._get_prompt(row[question_type])#PROMPTS[self.prompt_num].format(row[question_type])
                
                question_responses = []
                
                tot_undefined_answ = 0
                attempt = 0
                num_obtained_answers = 0
                #Until I have less than 5 answers and less than 5 attempts I keep asking
                while num_obtained_answers < MAX_NUM_ANSWERS and attempt < 5:
                    resp = self.model.call_model(full_prompt)
                    resp = self.check_binary_answer(resp)
                    if resp == UNDEFINED:
                        attempt = attempt + 1
                        tot_undefined_answ = tot_undefined_answ + 1
                        continue
                    question_responses.append(resp)
                    attempt = 0
                    num_obtained_answers = num_obtained_answers + 1
                if attempt == 5:
                    question_responses.append(UNDEFINED)
                    attempt = 0
                    num_obtained_answers = num_obtained_answers + 1
                logger.info(f"{subcategory} - {question_type}: {question_responses}")
                rainbow_meter[question_type].append(self.combine_binary_answers(question_responses))
                rainbow_meter[f"{VALIDITY} {question_type}"].append(prompt_validity_score(tot_undefined_answ))
                rainbow_meter[f"{COHERENCE} {question_type}"].append(coherence_score(question_responses))
                
            #Export Rainbow Meter
            rainbow_meter_df = pd.DataFrame(rainbow_meter)
            self.export_rainbow_meter(rainbow_meter_df)

    #Return True if the results exists, otherwise False
    def rainbow_meter_results_exist(self):
        result_path = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model.name}/"
        
        if self.scenario == SCENARIO_LANGUAGE:
            scenario_path = f"{self.language}_rainbow_meter_{self.prompt_num}.csv"
        elif self.scenario == SCENARIO_NATIONALITY:
            scenario_path = f"{self.language}_rainbow_meter_{self.prompt_num}.csv"
        
        if os.path.exists(result_path): #If exist
            csv = pd.read_csv(result_path, sep=";")
            num_rows = len(csv)
            if num_rows < CRITERIA_NUM:
                self.rainbow_meter_questions = pd.read_csv(f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model.name}/{self.country.language_code}_rainbow_meter_{self.prompt_num}.csv", sep=";")
                self.criteria_filled = num_rows
                return False #Return False as it is incomplete or absent
            return True #Return True as it is complete
        self.rainbow_meter_questions = get_rainbow_map(self.country.language_code)
        return False #Return False as it is incomplete or absent

    #Return yes/no/unsure/undefined based on the answer
    def check_binary_answer(self, response):
        response = response.lower().replace(".", "").replace("*", "").replace('"', '').strip()
        first_word = response.split()[0].strip(",;:!?.")
        if self.prompt_num == 0 and response and first_word in {YES, NO}: #Response is valid
                    return first_word
        elif self.prompt_num == 1 and response and first_word in {YES, NO, "unsure"}:
                    return first_word
        #Response is not valid
        return UNDEFINED
    
    #Combine the answers get for each criterion and combine them, returns the average score (yes=1, no=0, unsure=0.5) of the values and -1 if at least one "undefined" is present
    def combine_binary_answers(self, responses):
        if UNDEFINED in responses:
            return -1

        mapping = {
            YES: 1.0,
            NO: 0.0,
            "unsure": 0.5
        }

        try:
            values = [mapping[l] for l in responses]
        except KeyError as e:
            raise ValueError(f"Unexpected label found: {e}")

        return sum(values) / len(values)
    
    #Export and save the Rainbow Meter
    def export_rainbow_meter(self, rainbow_meter):
        result_path = f"{RESULT_PATH}/{RAINBOW_METER_PATH}/{self.scenario}/{self.model.name}"
        if self.scenario == SCENARIO_LANGUAGE:
            rainbow_meter.to_csv(f"{result_path}/{self.country.language}_rainbow_meter_{self.prompt_num}.csv", sep=";", index=False)
        elif self.scenario == SCENARIO_NATIONALITY:
            rainbow_meter.to_csv(f"{result_path}/{self.country.id}_rainbow_meter_{self.prompt_num}.csv", sep=";", index=False)
            
    def _get_prompt(self, question):
        if self.scenario == SCENARIO_LANGUAGE:
            PROMPTS = [
                """{0}
                    Provide a yes or no answer.""",
                """{0}
                    Provide a "yes", "no" or "unsure" answer.""", 
            ]
            tmp =  PROMPTS[self.prompt_num].format(question)
        elif self.scenario == SCENARIO_NATIONALITY:
            PROMPTS = [
                """{0}
                    Provide a yes or no answer as if you were a {1} citizen.""",
                """{0}
                    Provide a "yes", "no" or "unsure" answer as if you were a {1} citizen.""", 
            ]
            
            tmp =  PROMPTS[self.prompt_num].format(question, self.country.citizenship)
        return tmp

def get_rainbow_map(language_code):
    result_path = f"data/{RAINBOW_METER_PATH}/rainbow_meter_{language_code}.csv"
    return pd.read_csv(result_path, sep=";")

def rainbow_map_language_exist(language_code):
    file_path = f"data/{RAINBOW_METER_PATH}/rainbow_meter_{language_code}.csv"
    return os.path.exists(file_path)

def prompt_validity_score(undefined_count):
    num_tot_answ = 15
    return 1 - (undefined_count / num_tot_answ)

def coherence_score(answers):
        counts = {}
        # Count occurrences manually
        for a in answers:
            if a in counts:
                counts[a] += 1
            else:
                counts[a] = 1
        # Find the maximum count
        most_common = 0
        for count in counts.values():
            if count > most_common:
                most_common = count
        
        return most_common / len(answers)

