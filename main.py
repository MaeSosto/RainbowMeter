from lib.constants import *
from lib.models import *
from lib.country import *
from lib.rainbow_meter import *

error = False
#Iterate on each model
for model_name in [LLAMA3, LLAMA3_70B, GEMMA3, GPT4, GPT4_MINI, GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_LITE, DEEPSEEK_671B]:
    
    #Setup the model
    # model_name = LLAMA3
            
    #Get country file
    country_list = get_country_list()

    #Iterate on every country
    for country_name in country_list: 
        country = Country(country_list, country_name)
        
        #NOW ONLY EN MODEL SUPPORTED ENGLISH
        if country.language != 'en': #or country_info.check_language_file(model_name):
            continue
        
        #If results already exist for this model and language then skip 
        if check_result_already_exist(model_name, country.language):
            logger.info(f"✔️ {MODELS_LABELS[model_name]} - {country.country_id} - {country.language}")
            continue
        
        model = Model(model_name)
        error = model.initialize_model()
        if error: #If there are no errors in initializing the model
            break  
        
        rainbow_meter = Rainbow_Meter(model, country)
        logger.info(f"🔄 {MODELS_LABELS[model_name]} - {country.country_id} - {country.language}")
        rainbow_meter.get_answers()
        
        #break
    if error:
        break
