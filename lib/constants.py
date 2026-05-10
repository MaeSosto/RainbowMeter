import pandas as pd
import data
import logging
import torch
import json
import pdb
import tqdm
import os

# =============================
# Logging Configuration
# =============================
logger = logging.getLogger()
#logging.basicConfig(filename='.log', encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logger.addHandler(console)

# =============================
# Device Configuration
# =============================
if torch.backends.mps.is_available():
    device = torch.device('cpu')  # Temporarily using CPU over MPS
elif torch.cuda.is_available():   
    device = torch.device("cuda") 
else: 
    device = torch.device('cpu')
    
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))

torch.set_default_device(device)
logger.info(f"Using device: {device}")

#Rainbow Map
CATEGORY = "Category"
SUBCATEGORY = "Subcategory"
RAINBOW_MAP_DF = pd.read_csv("data/rainbow_map/rainbow_map.csv", delimiter=";", index_col="country_id")
CRITERIA_WEIGHTS_DF = pd.read_csv("data/rainbow_map/criteria_weights.csv", delimiter=";", index_col=SUBCATEGORY)
RAINBOW_MAP_CATEGORIES = ["Equality & non-discrimination", "Family", "Hate crime & hate speech", "Legal gender recognition", "Intersex bodily integrity", "Civil society space", "Asylum"]
with open("data/countries.json") as f:
    COUNTRIES_FILE = json.load(f)
TOT_CRITERIA_NUM = (RAINBOW_MAP_DF.shape[1])-2

#Countries
NAME = "name"
ID = "country_id"
LANGUAGES = "languages"
LANGUAGES_CODE = "languages_code"
CITIZENSHIP = "citizenships"

#Rainbow Meter
FACT = "Fact"
SUPPORT = "Support"
OPPOSITION = "Opposition"
STANCE = "Stance"
QUESTION_TYPES = [FACT, SUPPORT, OPPOSITION]
STANCE = "Stance"
YES = "yes"
NO = "no"
UNDEFINED = "undefined"

#Create Project Folder Structure
RESULT_PATH = "results"
os.makedirs(RESULT_PATH, exist_ok=True)
RAINBOW_METER_RESULT_PATH = f"{RESULT_PATH}/rainbow_meter"
os.makedirs(RAINBOW_METER_RESULT_PATH, exist_ok=True)
EVALUATIONS_PATH = f"{RESULT_PATH}/evaluations"
os.makedirs(EVALUATIONS_PATH, exist_ok=True)
GRAPHS_PATH = f"graphs"
os.makedirs(GRAPHS_PATH, exist_ok=True)
RAINBOW_METER_DATA_PATH = "data/rainbow_meter"


SCENARIO_LANGUAGE = "language_scenario"
SCENARIO_NATIONALITY = "nationality_scenario"
SCENARIO_LAN_NAT = "language_nationality_scenario"
SCENARIOS = [SCENARIO_LANGUAGE, SCENARIO_NATIONALITY, SCENARIO_LAN_NAT]
MODELS_PERFORMANCES_PATH = "models_performances"
for s in SCENARIOS:
    os.makedirs(f"{RAINBOW_METER_RESULT_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{EVALUATIONS_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{GRAPHS_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{GRAPHS_PATH}/{MODELS_PERFORMANCES_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{EVALUATIONS_PATH}/{MODELS_PERFORMANCES_PATH}/{s}", exist_ok=True)
    
RAINBOW_METER_EN = pd.read_csv(f"{RAINBOW_METER_DATA_PATH}/{SCENARIO_LANGUAGE}/rainbow_meter_en.csv", sep=";", index_col=SUBCATEGORY)

### MODEL LIST ###
QWEN35_2 = "Qwen/Qwen3.5-2B" #HF
QWEN35_9 = "Qwen/Qwen3.5-9B" #HF
QWEN35_27 = "Qwen/Qwen3.5-27B" #HF
LLAMA32_3 = "meta-llama/Llama-3.2-3B" #HF
LLAMA31_8 = "meta-llama/Llama-3.1-8B" #HF
LLAMA31_70 = "meta-llama/Llama-3.1-70B" #HF
DEEPSEEKV32 = "deepseek-chat"
SONNET46 = "claude-sonnet-4-6"
GPT54 = 'gpt-5.4'
GPT54_MINI = 'gpt-5.4-mini'
GEMINI3_FLASH = 'gemini-3-flash-preview'
DEEPL = "DeepL" 

MODEL_LIST = [QWEN35_2, QWEN35_9, QWEN35_27, LLAMA32_3, LLAMA31_8, LLAMA31_70, DEEPSEEKV32, SONNET46, GPT54, GEMINI3_FLASH]

MODEL_LABEL = {
    QWEN35_2: "Qwen3.5 2B",
    QWEN35_9: "Qwen3.5 9B",
    QWEN35_27: "Qwen3.5 27B",
    LLAMA32_3: "LlaMa 3.2 3B",
    LLAMA31_8: "LlaMa 3.1 8B",
    LLAMA31_70: "LlaMa 3.1 70B",
    DEEPSEEKV32: "DeepSeek-V3.2",
    SONNET46: "Sonnet 4.6",
    DEEPL: "DeepL",
    GPT54 : 'GPT 5.4',
    GPT54_MINI : 'GPT 5.4-mini',
    GEMINI3_FLASH: 'Gemini 3 Flash'
}

#Return True if the results exists, otherwise False
def get_rainbow_meter_file_answers(scenario, model_name, language_code = "", country_id = ""):
    result_path = f"{RAINBOW_METER_RESULT_PATH}/{scenario}/{model_name}/"
    if scenario == SCENARIO_LANGUAGE:
        scenario_path = f"rm_answers_{language_code}.csv"
    elif scenario == SCENARIO_NATIONALITY:
        scenario_path = f"rm_answers_{country_id}.csv"
    else:
        scenario_path = f"rm_answers_{language_code}_{country_id}.csv"
    if os.path.exists(result_path+scenario_path):
        
        df = pd.read_csv(result_path+scenario_path, sep=";", index_col=SUBCATEGORY) 
        if df.empty:
            logger.error(f"The rainbow meter {result_path+scenario_path} is empty")
            return pd.DataFrame()
        if df.shape[0] == TOT_CRITERIA_NUM:
            return df
        else:
            logger.error(f"⚠️ {result_path+scenario_path} is incomplete")
    else:
        logger.error(f"⚠️ {result_path+scenario_path} is missing")
    return pd.DataFrame()

#Get the Rainbow Meter file based on the scenario
def get_rainbow_meter_file_default(scenario, language_code, country_id):
    result_path = f"{RAINBOW_METER_DATA_PATH}/{scenario}/"
    if scenario == SCENARIO_LANGUAGE:
        scenario_path = f"rainbow_meter_{language_code}.csv"
    elif scenario == SCENARIO_NATIONALITY:
        scenario_path = f"rainbow_meter_{country_id}.csv"
    else:
        scenario_path = f"rainbow_meter_{language_code}_{country_id}.csv"
    if os.path.exists(result_path+ scenario_path): #If exist
        df = pd.read_csv(result_path+scenario_path, sep=";", index_col=SUBCATEGORY)
        return True,  df
    logger.error(f"⚠️ {result_path+scenario_path} is missing")
    return False, None