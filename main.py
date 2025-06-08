from lib import *
from pathlib import Path
from models import callModel

countries_file = "data/countries_langs.json"
countries_file = open(countries_file)

# returns JSON object as a dictionary
countries_file = json.load(countries_file)

modelName = 'llama3'
#modelName = 'gemma3'
path_result = f"results_for_analysis/languages_experiments/{modelName}"
Path(path_result).mkdir(parents=True, exist_ok=True)

for country_name in countries_file:
    country_info = countries_file[country_name]
    lan = country_info['languages_code'][0]
    #lan = country.languages_code
    country_id = country_info['COUNTRY_ID']
    #with open('data/criteria.json') as criteria_file:   
    criteria_file = f'data/prompts/prompt_{lan}.json'
    prompts_template = f'data/prompts_data/prompts_template_{lan}.json'
    #prompt_file = f'prompts_data/prompts_template_{lan}.json
    # Opening JSON file
    criteria_file = open(criteria_file)

    # returns JSON object as a dictionary
    criteria_file = json.load(criteria_file)

    promptList = [v for v in criteria_file if v['Question'] != ""]
    callModel(
        path_result = path_result,
        prompts_template = prompts_template
        prompt_info = promptList, 
        modelName = modelName, 
        lan = lan,
        country_id = country_id
    )



