from lib.constants import *
from lib.models import *
from lib.prompt import *
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# def _get_short_prompt_type(prompt_type):
#     return prompt_type.split()[1].lower()

def _get_country_info(country_name, model_name, prompt_type = "Question Pro"):
    country_info = countries_file[country_name]
    lan = country_info['languages_code'][0]
    country_id = country_info['COUNTRY_ID']

    prompt_template = f'data/prompts_data/prompts_template_{lan}.json'
    prompt_template = open(prompt_template)
    prompt_template = json.load(prompt_template)

    criteria_file = f'data/rainbow_meter/rainbow_meter_{lan}.json'
    criteria_file = open(criteria_file)
    criteria_file = json.load(criteria_file)
    criteriaList = [v for v in criteria_file if prompt_type in v and v[prompt_type] != ""]
    
    path_result = f"results_for_analysis/languages_experiments/{model_name}"
    Path(path_result).mkdir(parents=True, exist_ok=True)
    file_out = f"{path_result}/{lan}_{country_id}_raibow_meter.csv"
    return criteriaList, prompt_template, file_out

countries_file = "data/countries_langs.json"
countries_file = open(countries_file)
countries_file = json.load(countries_file)

#ONLY OLLAMA MODELS ARE NOW SUPPORTED
#model_name = 'llama3'
model_name = 'gemma3'
prompt_types = ['Question Pro', 'Question Con', 'Question Op']

#Iterate on every country
for country_name in countries_file:
    country_info = countries_file[country_name]
    language = country_info['languages_code'][0]
    if language != 'en': #NOW ONLY EN MODEL SUPPORTED ENGLISH
        continue
    
    criteriaList, prompt_template, file_out = _get_country_info(country_name, model_name)

    results = []
    for criteria in tqdm(criteriaList):
        
        risp = {
            'Category': criteria['Category'],
            'Subcategory': criteria['Subcategory']
        }
        
        for prompt_type in prompt_types:
            question = criteria[prompt_type]
            prompt = get_standard_prompt(prompt_template, question)

            response = call_model(
                prompt = prompt, 
                modelName = model_name, 
            )
            
            risp[prompt_type] = response
        
        results.append(risp)
        json_object = json.dumps(results, indent=4)

        with open(file_out, "w") as outfile:
            outfile.write(json_object)
        



