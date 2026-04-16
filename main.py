from lib.constants import *
from lib.models import *
from lib.prompting import *
from lib.evaluations import *


#PROMPT TYPES
#0 = Provide a yes or no answer.
#1 = Provide a "yes", "no" or "unsure" answer.
PROMPT_NUM = 0

#Scenario
#scenario = SCENARIO_LANGUAGE
scenario = SCENARIO_NATIONALITY

#Setup the model
model_name = QWEN3_4
model = Model(model_name)
error = model.initialize_model()
if not error: #If there are no errors in initializing the model
            
    rainbow_meter = Rainbow_Meter(model, scenario, PROMPT_NUM)
    rainbow_meter.get_answers()
    
    #Evaluations
    logger.info("🧮 Evaluations")
    eval = Evaluations(model, scenario, PROMPT_NUM)
    # eval.calculate_wilcoxon()
    

logger.info("✅ Done")