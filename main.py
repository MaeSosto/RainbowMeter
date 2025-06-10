from lib.constants import *
from lib.models import *
from lib.prompt import *
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def _get_country_info(country_name, model_name):
    country_info = countries_file[country_name]
    lan = country_info['languages_code'][0]
    country_id = country_info['COUNTRY_ID']

    prompt_template = f'data/prompts_data/prompts_template_{lan}.json'
    prompt_template = open(prompt_template)
    prompt_template = json.load(prompt_template)

    criteria_file = f'data/rainbow_meter/rainbow_meter_{lan}.json'
    criteria_file = open(criteria_file)
    criteria_file = json.load(criteria_file)
    criteriaList = [v for v in criteria_file if v['Question'] != ""]
    
    path_result = f"results_for_analysis/languages_experiments/{model_name}"
    Path(path_result).mkdir(parents=True, exist_ok=True)
    file_out = f"{path_result}/{model_name}-{lan}_{country_id}_raibow_meter.csv"
    return criteriaList, prompt_template, file_out

countries_file = "data/countries_langs.json"
countries_file = open(countries_file)
countries_file = json.load(countries_file)

#modelName = 'gpt-4.1-mini'

#ONLY OLLAMA MODELS ARE NOW SUPPORTED
#modelName = 'llama3'
model_name = 'gemma3'

#Iterate on every country
for country_name in countries_file:
    country_info = countries_file[country_name]
    language = country_info['languages_code'][0]
    if language != 'en': #NOW ONLY EN MODEL SUPPORTED ENGLISH
        continue
    criteriaList, prompt_template, file_out = _get_country_info(country_name)

    results = []
    for criteria in tqdm(criteriaList):

        prompt = get_standard_prompt(prompt_template, criteria['Question'])

        response = call_model(
            prompt = prompt, 
            modelName = model_name, 
        )
        
        results.append([
            criteria['Category'],
            criteria['Subcategory'],
            criteria['Question'],
            response
        ])
        #df = pd.DataFrame(results, columns=['text','label_0','label_1','label_2','label_3','label_4','majority_label','majority_label_postprocessed'])
        df = pd.DataFrame(results, columns=['Caegory','Subcategory','Prompt', 'Response'])
        df.to_csv(file_out, index_label='index')
        



