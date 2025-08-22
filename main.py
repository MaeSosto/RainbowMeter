from lib.constants import *
from lib.models import *
from lib.country import *


NUM_ATTEMPT = 5
NUM_ANSWERS = 1

for model_name in MODEL_LIST:
    
    #Setup the model
    # model_name = LLAMA3
    model = Model(model_name)
    error = model.initialize_model()
    if not error: #If there are no errors in initializing the model
            
        #Get country file
        country_list = get_country_list()

        #Iterate on every country
        for country_name in country_list: 
            country_info = Country(country_list[country_name])
            country_language = country_info.get_language()
            
            #NOW ONLY EN MODEL SUPPORTED ENGLISH
            if country_language != 'en': #or country_info.check_language_file(model_name):
                continue
            
            #Get the criteria file
            criteria_list = country_info.get_criteria_file()
            
            #If results already exist for this model and language then skip 
            if country_info.check_result_already_exist(model_name):
                logger.info(f"✔️ {MODELS_LABELS[model_name]} - {country_info.get_country_id()} - {country_language}")
                continue
            
            logger.info(f"🔄 {MODELS_LABELS[model_name]} - {country_info.get_country_id()} - {country_language}")

            results = []
            #Iterate on the criteria list
            for criteria in tqdm(criteria_list):
                
                logger.info(criteria['Subcategory'])
                risp = {
                    'Category': criteria['Category'],
                    'Subcategory': criteria['Subcategory'],
                }
                
                #Iterate on the prompt types
                for prompt_type in prompt_types:
                    logger.info(prompt_type)
                    risp[prompt_type] = []
                    
                    #Try with standard prompt    
                    attempt = 0
                    prompt = country_info.get_standard_prompt(criteria[prompt_type])
                    
                    #Try to get a NUM_ANSWERS ok answers until NUM_ATTEMPTS attempts
                    while attempt < NUM_ATTEMPT and len(risp[prompt_type]) < NUM_ANSWERS: 
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
                            prompt = country_info.get_retry_prompt(criteria[prompt_type], prompt + "\n" + str(response))
                    
                    #After 5 attempts return error
                    if attempt == NUM_ATTEMPT:
                        logger.error(f"Error: {prompt_type}" )
                        risp[prompt_type] = None
                results.append(risp)
                
                country_info.export_language_results(results, model_name)
