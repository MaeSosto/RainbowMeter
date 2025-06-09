from lib import *
from pathlib import Path
from models import callModel, _getPrompt

def _get_country_info(country_name):
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
    
    file_out = f"{path_result}/{modelName}-{lan}_{country_id}_raibow_meter.csv"
    return criteriaList, prompt_template, file_out

countries_file = "data/countries_langs.json"
countries_file = open(countries_file)

# returns JSON object as a dictionary
countries_file = json.load(countries_file)

#modelName = 'llama3'
modelName = 'gemma3'
path_result = f"results_for_analysis/languages_experiments/{modelName}"
Path(path_result).mkdir(parents=True, exist_ok=True)


for country_name in countries_file:
    criteriaList, prompt_template, file_out = _get_country_info(country_name)
    results = []
    for criteria in tqdm(criteriaList):
        subcategory = criteria['Subcategory']
        prompt = _getPrompt(prompt_template, criteria['Question'])

        response = callModel(
            prompt = prompt, 
            modelName = modelName, 
        )
        #print(response)
        
        results.append(
            [
                criteria['Category'],
                criteria['Subcategory'],
                criteria['Question'],
                response
            ]
        )
        #df = pd.DataFrame(results, columns=['text','label_0','label_1','label_2','label_3','label_4','majority_label','majority_label_postprocessed'])
        df = pd.DataFrame(results, columns=['Ctegory','Subcategory','Prompt', 'Response'])
        df.to_csv(file_out, index_label='index')
        



