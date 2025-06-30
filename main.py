from lib.constants import *
from lib.models import *
from lib.prompt import *



countries_file = get_country_file()
model_name = LLAMA3

model = Model(model_name)

#Iterate on every country
for country_name in countries_file: 
    country_info = Prompt(countries_file[country_name])
    
    language = country_info.get_language()
    if language != 'en': #or country_info.check_language_file(model_name): #NOW ONLY EN MODEL SUPPORTED ENGLISH
        continue
    
    criteria_list = country_info._get_criteria_file()

    results = []
    for criteria in tqdm(criteria_list):
        
        logger.info(criteria['Subcategory'])
        risp = {
            'Category': criteria['Category'],
            'Subcategory': criteria['Subcategory'],
        }
        
        for prompt_type in prompt_types:
            logger.info(prompt_type)
            risp[prompt_type] = []
            
            #Try with standard prompt    
            attempt = 0
            prompt = country_info.get_standard_prompt(criteria[prompt_type])
            while attempt < 5 and len(risp[prompt_type]) < 5: 
                response = model.call_model(prompt)
                clean_response = country_info.check_response(response)

                if clean_response != None: #Response is valid
                    risp[prompt_type].append(clean_response)
                    logger.info(f"{attempt} | {risp[prompt_type]}")
                    attempt = 0
                    prompt = country_info.get_standard_prompt(criteria[prompt_type])
                else:
                    logger.info(f"{attempt} | {response}")
                    #Try with retry prompt
                    attempt += 1
                    prompt = country_info.get_retry_prompt(criteria[prompt_type], prompt + "\n" + response)
            if attempt == 5:
                logger.error(f"Error: {prompt_type}" )
                risp[prompt_type] = None
        results.append(risp)
        country_info.export_language_results(results, model_name)
        
        



