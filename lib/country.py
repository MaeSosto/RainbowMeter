import re
import json
import os

folder_result = "results_for_analysis/"
folder_language_scenario = "language_scenario/"

def get_country_list():
        countries_file = "data/countries.json"
        countries_file = open(countries_file)
        countries_file = json.load(countries_file)
        return countries_file
        
class Country:
    def __init__(self, country_list, country_name):
        self.country_name = country_name
        self.country_id = country_list[country_name]["COUNTRY_ID"]
        self.language = country_list[country_name]['languages_code'][0]
        self.labels = labels
    def check_language_file(self, model_name):
        path_result = folder_result+ folder_language_scenario+model_name 
        file_out = f"{path_result}/{self.language}_raibow_meter.csv"
        return os.path.isfile(file_out)
        


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