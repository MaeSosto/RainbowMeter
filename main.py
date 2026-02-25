from lib.constants import *
from lib.models import *
from lib.rainbow_meter import *
from lib.evaluations import *


#PROMPT TYPES
#0 = Provide a yes or no answer.
#1 = Provide a "yes", "no" or "unsure" answer.
PROMPT_NUM = 0

#Scenario
#scenario = SCENARIO_LANGUAGE
scenario = SCENARIO_NATIONALITY

#Setup the model
model_name = LLAMA3
model = Model(model_name)
error = model.initialize_model()
if not error: #If there are no errors in initializing the model
        
    #Get country file
    #country_list = get_country_list()

    #Iterate on every country
    for country_name in COUNTRIES_FILE: 
        
        #Iterate on every language and citizenship 
        for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
            
            if language != 'en': #NOW ONLY EN MODEL SUPPORTED ENGLISH
                continue
            country = Country(country_name, country_identity_num)
            
            rainbow_meter = Rainbow_Meter(model, country, scenario, PROMPT_NUM)
            rm_correct = rainbow_meter.rainbow_meter_results_exist(country)
            if not rm_correct:
                rainbow_meter.get_answers()
    
    #Evaluations
    # logger.info("🧮 Evaluations")
    # eval = Evaluations(model, PROMPT_NUM, scenario)
    # eval.get_evaluations()
    

logger.info("✅ Done")