import re
import json
import os

prompt_types = ['Statement Pro', 'Statement Con'] #, 'Question Op'] #Let's stay with just this now 

def get_country_list():
        countries_file = "data/countries_langs.json"
        countries_file = open(countries_file)
        countries_file = json.load(countries_file)
        return countries_file
        
class Country:
    def __init__(self, country_info):
        self.country_info = country_info
        self.country_id = country_info["COUNTRY_ID"]
        #self.country_name = country_info
        self.language = self.country_info['languages_code'][0]
        if self.language == "en": #NOW ONLY EN MODEL SUPPORTED ENGLISH
            self.prompt_template, self.retry_prompt_template, self.labels = self._get_prompt_templates()
            self.criteria_file = self._get_criteria_file() #NOW RETURN ONLY COMPLETE CRITERIAS 
        
    def get_language(self):
        return self.language
    
    def get_country_id(self):
        return self.country_id
    
    def get_criteria_file(self):
        return self.criteria_file

    def _get_prompt_templates(self):
        prompt_template = f'data/prompts_data/prompts_template_{self.language}.json'
        prompt_template = open(prompt_template)
        prompt_template = json.load(prompt_template)
        return prompt_template["prompt_template"], prompt_template["retry_prompt_template"], prompt_template["labels"]
    
    def _get_criteria_file(self):
        criteria_file = f'data/rainbow_meter/rainbow_meter_{self.language}.json'
        criteria_file = open(criteria_file)
        criteria_file = json.load(criteria_file)
        
        #NOW RETURN ONLY COMPLETE CRITERIAS 
        criteria_file = [criteria for criteria in criteria_file if criteria[prompt_types[0]] != ""] 
        return criteria_file
    
    def check_result_already_exist(self, model_name):
        path_result = f"results_for_analysis/languages_experiments/{model_name}"
        file_out = f"{path_result}/{self.language}_raibow_meter.csv"
        if os.path.exists(file_out):
            file_out = open(file_out)
            file_out = json.load(file_out)
            
            if len(file_out) == len(self.criteria_file):
                return True
        return False 

    def export_language_results(self, results, model_name):
        json_object = json.dumps(results, indent=4)
        path_result = f"results_for_analysis/languages_experiments/{model_name}"
        os.makedirs(path_result, exist_ok=True)
        file_out = f"{path_result}/{self.language}_raibow_meter.csv"
        with open(file_out, "w") as outfile:
            outfile.write(json_object)
    
    def check_language_file(self, model_name):
        path_result = f"results_for_analysis/languages_experiments/{model_name}"
        file_out = f"{path_result}/{self.language}_raibow_meter.csv"
        return os.path.isfile(file_out)
        
    def get_standard_prompt(self, text):
        format_instructions = f'{{ "answer": "{self.labels[0]}" }}'

        prompt_template = self.prompt_template.format(
            text=text,
            format_instructions=format_instructions
        )
        return prompt_template

    def get_retry_prompt(self, text, chat_history):
        format_instructions = f'{{ "answer": "{self.labels[0]}" }}'

        prompt_template = self.retry_prompt_template.format(
            chat_history = chat_history,
            text=text,
            format_instructions=format_instructions
        )
        return prompt_template
    
    def check_response(self, response):
        try:
            response = re.sub(r"^```(?:json)?\n|```$", "", response.strip())
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = json.loads(response)
            if response["answer"] in self.labels:
                return response["answer"]
            else:
                return None
        except Exception as X:
            return None