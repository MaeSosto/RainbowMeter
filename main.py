from lib.constants import *
from lib.models import *
from lib.rainbow_meter import *

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

    #Iterate on every country
    for country in country_list: 
        
        #NOW ONLY EN MODEL SUPPORTED ENGLISH
        if country[LANGUAGES_CODE][0] != 'en':
            continue
        
        exist, criteria_filled = rainbow_meter_exist(model_name, country[LANGUAGES_CODE][0], PROMPT_NUM)
        if not exist or criteria_filled < CRITERIA_NUM:
            raibow_meter = Rainbow_Meter(model, country, PROMPT_NUM, criteria_filled)

logger.info("✅ Done")