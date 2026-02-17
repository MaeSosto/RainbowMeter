from lib.constants import *
from lib.country import *

NUM_ANSWERS = 5 #Num answer we want for each criterion-stance


PROMPTS = [
    """{0}
        Provide a yes or no answer.""",
    """{0}
        Provide a "yes", "no" or "unsure" answer.""", 
]

class Rainbow_Meter:
    def __init__(self, model, country, prompt_num, criteria_filled = 0):
        self.prompt_num = prompt_num
        self.model = model
        self.country = Country(country)
        self.criteria_file = self.country.criteria_file
        logger.info(f"🔄 {MODELS_LABELS[self.model.model_name]} - {self.country.id} - {self.country.language}")
        
        rainbow_meter_row = {
            CATEGORY: [],
            SUBCATEGORY: [],
            QUESTION_FACT: [], 
            QUESTION_SUPPORT: [], 
            QUESTION_OPPOSITION: [],
        }
        
        #If the csv contains answers already, then fill it up until there and continue from there
        for subcategory, row in self.criteria_file[:criteria_filled].iterrows():
            rainbow_meter_row[CATEGORY].append(row[CATEGORY])
            rainbow_meter_row[SUBCATEGORY].append(subcategory)
            rainbow_meter_row[QUESTION_FACT].append(row[QUESTION_FACT])
            rainbow_meter_row[QUESTION_SUPPORT].append(row[QUESTION_SUPPORT])
            rainbow_meter_row[QUESTION_OPPOSITION].append(row[QUESTION_OPPOSITION])
            
        
        #Get answers for the missing criterion in the csv file
        for subcategory, row in self.criteria_file[criteria_filled:].iterrows():
            rainbow_meter_row[CATEGORY].append(row[CATEGORY])
            rainbow_meter_row[SUBCATEGORY].append(subcategory)
            
            #Iterate on the prompt types
            for question_type in QUESTION_TYPES:
                full_prompt = PROMPTS[self.prompt_num].format(row[question_type])
                
                responses = []
                attempt = 0
                x = 0
                while x < NUM_ANSWERS and attempt < 5:
                    resp = self.model.call_model(full_prompt)
                    resp = self.check_binary_answer(resp)
                    if resp == UNDEFINED:
                        attempt = attempt + 1
                        continue
                    attempt = 0
                    responses.append(resp)
                    x = x + 1
                if attempt == 5:
                    responses.append(UNDEFINED)
                    x = x + 1
                logger.info(f"{subcategory} - {question_type}: {responses}")
                rainbow_meter_row[question_type].append(self.combine_binary_answers(responses))
                
            #Export Rainbow Meter
            rainbow_meter_df = pd.DataFrame(rainbow_meter_row)
            self.export_rainbow_meter(rainbow_meter_df)

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
    def export_rainbow_meter(self, rainbow_meter_df):
            result_path = f"{RESULT_PATH}/{SCENARIO_LANGUAGE_PATH}/{self.model.model_name}/"
            os.makedirs(result_path, exist_ok=True)
            rainbow_meter_df.to_csv(f"{result_path}{self.country.language}_rainbow_meter_{self.prompt_num}.csv", index=False)

#Return True if the results exists, otherwise False
def rainbow_meter_exist(model_name, language, prompt_num):
    result_path = f"{RESULT_PATH}/{SCENARIO_LANGUAGE_PATH}/{model_name}/{language}_rainbow_meter_{prompt_num}.csv"
    if os.path.exists(result_path):
        num_rows = len(pd.read_csv(result_path))
        return True, num_rows
    return False, 0
