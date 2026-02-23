from lib.constants import *
from lib.models import *
from lib.rainbow_meter import *
from lib.evaluations import *

model_name = LLAMA3

#PROMPT TYPES
#0 = Provide a yes or no answer.
#1 = Provide a "yes", "no" or "unsure" answer.
PROMPT_NUM = 1

#Setup the model
model = Model(model_name)
error = model.initialize_model()
if not error: #If there are no errors in initializing the model
        
    #Get country file
    country_list = get_country_list()

    country_with_results = []
    #Iterate on every country
    for country in country_list: 
        country = Country(country)
        #NOW ONLY EN MODEL SUPPORTED ENGLISH
        if country.language != 'en':
            continue
        
        exist, criteria_filled = rainbow_meter_exist(model_name, country.language, PROMPT_NUM)
        if not exist or criteria_filled < CRITERIA_NUM:
            rainbow_meter = Rainbow_Meter(model, country, PROMPT_NUM)
            rainbow_meter.get_answers(criteria_filled)
        country_with_results.append(country)
        #evaluations = get_evaluations()
    
    #Evaluations
    logger.info("🧮 Evaluations")
    get_evaluations(model_name, PROMPT_NUM, country_with_results)
    

logger.info("✅ Done")