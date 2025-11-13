from lib.constants import *
from lib.models import *
from lib.country import *
from lib.rainbow_meter import *

#Iterate on each model
for model_name in [LLAMA3, LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK]:#, GPT4, GPT4_MINI, GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_LITE ]:
    
    #Setup the model
    # model_name = LLAMA3
    model = Model(model_name)
    error = model.initialize_model()
    if not error: #If there are no errors in initializing the model
            
        #Get country file
        country_list = get_country_list()

        #Iterate on every country
        for country_name in country_list: 
            country = Country(country_list, country_name)
            
            #NOW ONLY EN MODEL SUPPORTED ENGLISH
            if country.language != 'en': #or country_info.check_language_file(model_name):
                continue
            
            rainbow_meter = Rainbow_Meter(model, country)
            
            # #Get the criteria file
            # criteria_list = rainbow_meter.get_criteria_file()
            
            #If results already exist for this model and language then skip 
            if rainbow_meter.check_result_already_exist():
                logger.info(f"✔️ {MODELS_LABELS[model_name]} - {country.country_id} - {country.language}")
                continue
            
            rainbow_meter.get_answers()
            logger.info(f"🔄 {MODELS_LABELS[model_name]} - {country.country_id} - {country.language}")
            #break
