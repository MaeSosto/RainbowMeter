from lib.constants import *
#import re
from collections import OrderedDict

NAME = "name"
ID = "country_id"
LANGUAGES = "languages"
LANGUAGES_CODE = "languages_code"
CITIZENSHIP = "citizenships"

#Access the country file and get the json values of each country in 
# def get_country_list():
#         countries_file = COUNTRIES_FILE
#         countries_file = open(countries_file)
#         countries_file = json.load(countries_file)
#         return countries_file


# def get_countries_specific_language(language):
#     countries_selection = COUNTRIES_FILE.copy()
    
#     for country in COUNTRIES_FILE:
#         if country[LANGUAGES_CODE][0] == language:
#             countries_selection.append(country)
    
#     #Get country list
#     country_list = get_country_list()
    
#     #Iterate on every country
#     for country in country_list:
#         if country[LANGUAGES_CODE][0] == language:
#             countries_selection.append(country)
#     return countries_selection

class Country:
    def __init__(self, country_name, country_identity_num):
        self.name = country_name
        self.id = COUNTRIES_FILE[country_name][ID]
        self.language = COUNTRIES_FILE[country_name][LANGUAGES][country_identity_num]
        self.language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
        self.citizenship = COUNTRIES_FILE[country_name][CITIZENSHIP][country_identity_num]
        # if self.language == "en": #NOW ONLY EN MODEL SUPPORTED ENGLISH
        #     self.rainbow_meter_questions = pd.read_csv(f'{RESULT_PATH}{RAINBOW_METER_PATH}/rainbow_meter_{self.language_code}.csv', delimiter=";", index_col="Subcategory") 
            #self.prompt_template, self.retry_prompt_template, self.labels = self._get_prompt_templates()

#Get the list of countries that speaks the specified language
def get_countries_that_speaks(language = "en"):
    countries = OrderedDict()
    
    for country_name in COUNTRIES_FILE: 
        
        #Iterate on every language and citizenship 
        if language in COUNTRIES_FILE[country_name][LANGUAGES]:
            countries[country_name] = {
                "country_id": COUNTRIES_FILE[country_name][ID],
                "citizenships": COUNTRIES_FILE[country_name][CITIZENSHIP],
                "languages": COUNTRIES_FILE[country_name][LANGUAGES],
                "languages_code": COUNTRIES_FILE[country_name][LANGUAGES_CODE],
            }
    return countries
    
    # def _get_prompt_templates(self):
    #     prompt_template = f'data/prompts_data/prompts_template_{self.language}.json'
    #     prompt_template = open(prompt_template)
    #     prompt_template = json.load(prompt_template)
    #     return prompt_template["prompt_template"], prompt_template["retry_prompt_template"], prompt_template["labels"]
    
    # def get_criteria_file(self):
    #     criteria_file = pd.read_csv(f'data/rainbow_meter/rainbow_meter_{self.language}.csv', delimiter=",")
        
        
    #     #NOW RETURN ONLY COMPLETE CRITERIAS 
    #     criteria_file = [criteria for criteria in criteria_file if criteria[prompt_types[0]] != ""] 
    #     return criteria_file
    
    # def check_result_already_exist(self, model_name):
    #     file_out = RESULT_PATH+SCENARIO_LANGUAGE_PATH+f"{self.language}_raibow_meter.csv"
    #     if os.path.exists(file_out):
    #         file_out = open(file_out)
    #         file_out = json.load(file_out)
            
    #         if len(file_out) == len(self.criteria_file):
    #             return True
    #     return False 

    # def export_language_results(self, results, model_name):
    #     json_object = json.dumps(results, indent=4)
    #     path_result = f"results/languages_scenario/{model_name}"
    #     os.makedirs(path_result, exist_ok=True)
    #     file_out = f"{path_result}/{self.language_code}_raibow_meter.csv"
    #     with open(file_out, "w") as outfile:
    #         outfile.write(json_object)
    
    # def check_language_file(self, model_name):
    #     path_result = f"results/languages_scenario/{model_name}"
    #     file_out = f"{path_result}/{self.language_code}_raibow_meter.csv"
    #     return os.path.isfile(file_out)
        
    # def get_standard_prompt(self, text):
    #     format_instructions = f'{{ "answer": "{self.labels[0]}" }}'

    #     prompt_template = self.prompt_template.format(
    #         text=text,
    #         format_instructions=format_instructions
    #     )
    #     return prompt_template

    # def get_retry_prompt(self, text, chat_history):
    #     format_instructions = f'{{ "answer": "{self.labels[0]}" }}'

    #     prompt_template = self.retry_prompt_template.format(
    #         chat_history = chat_history,
    #         text=text,
    #         format_instructions=format_instructions
    #     )
    #     return prompt_template
    
    # def check_response(self, response):
    #     try:
    #         response = re.sub(r"^```(?:json)?\n|```$", "", response.strip())
    #         response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #         response = json.loads(response)
    #         if response["answer"] in self.labels:
    #             return response["answer"]
    #         else:
    #             return None
    #     except Exception as X:
    #         return None