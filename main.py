from lib.constants import *
from lib.models import *
from lib.country import *
from lib.rainbow_meter import *

error = False
#Iterate on each model
for model_name in [LLAMA3]:
            
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
            #if not check_binary_answers(model_name, country.language):
            rainbow_meter = Rainbow_Meter(model_name, country)
            #rainbow_meter.process_binary_answers()
            continue
        
    
        
        rainbow_meter = Rainbow_Meter(model_name, country)
        logger.info(f"🔄 {MODELS_LABELS[model_name]} - {country.country_id} - {country.language}")
        err = rainbow_meter.get_answers()
        if error: #If there are no errors in initializing the model
            break  
    if error:
        break
