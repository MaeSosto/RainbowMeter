import data
import logging
import torch
import json
import os
import re

# Logging Configuration
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Device Configuration
if torch.backends.mps.is_available():
    device = torch.device('cpu')  # Temporarily using CPU over MPS
elif torch.cuda.is_available():   
    device = torch.device("cuda") 
else: 
    device = torch.device('cpu')
    

torch.set_default_device(device)
logger.info(f"Using device: {device}")

#Rainbow Map
CRITERION_ID = "criterion_id"
RAINBOW_MAP = "Rainbow Map"
CATEGORY = "Category"
SUBCATEGORY = "Subcategory"
with open("data/countries.json") as f:
    COUNTRIES_FILE = json.load(f)
RAINBOW_MAP_CATEGORIES = ["Equality & non-discrimination", "Family", "Hate crime & hate speech", "Legal gender recognition", "Intersex bodily integrity", "Civil society space", "Asylum"]

#Countries
NAME = "name"
ID = "country_id"
LANGUAGES = "languages"
LANGUAGES_CODE = "languages_code"

#Evaluations and Grapohs
COH = "Coherence"
VAL = "val" 
COH_VAL = "coh-val"
EVALUATIONS_PATH = "evaluations"
MAE = "MAE"
PERCENTAGE = "percentage"
GRAPHS_PATH = "graphs"
AVG = "avg"

#Rainbow Meter
FACT = "Fact"
SUPPORT = "Support"
OPPOSITION = "Opposition"
STANCE = "Stance"
QUESTION_TYPES = [FACT, SUPPORT, OPPOSITION]
YES = "yes"
NO = "no"
UNDEFINED = "undefined"
RAINBOW_MAP = "Rainbow Map"
COUNTRY = "country"
LANGUAGE = "language"
MODEL = "model"
SCENARIO = "scenario"

LAN_SCENARIO = "lan"
COU_SCEANRIO = "cou"
LAN_COU_SCENARIO = "lan_cou"
SCENARIOS = [LAN_SCENARIO, COU_SCEANRIO, LAN_COU_SCENARIO]
SCENARIO_LABELS = {
    LAN_SCENARIO: "Language",
    COU_SCEANRIO: "Country",
    LAN_COU_SCENARIO: "Language-Country"
}

#Create Project Folder Structure
RAINBOW_METER = "rainbow_meter"
os.makedirs(RAINBOW_METER, exist_ok=True)
for s in SCENARIOS:
    os.makedirs(f"{RAINBOW_METER}/{s}", exist_ok=True)

RAINBOW_METER_DATA_PATH = "data/rainbow_meter"

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
GEMINI3_FLASH = 'gemini-3-flash-preview'
DEEPL = "DeepL" 

MODEL_LIST = [QWEN35_2, QWEN35_9, QWEN35_27, LLAMA32_3, LLAMA31_8, LLAMA31_70, DEEPSEEKV32, SONNET46, GPT54, GEMINI3_FLASH]

MODEL_LABEL = {
    QWEN35_2: "Qwen 2B",
    QWEN35_9: "Qwen 9B",
    QWEN35_27: "Qwen 27B",
    LLAMA32_3: "LLaMA 3B",
    LLAMA31_8: "LLaMA 8B",
    LLAMA31_70: "LLaMA 70B",
    DEEPSEEKV32: "DeepSeek",
    SONNET46: "Sonnet",
    DEEPL: "DeepL",
    GPT54 : 'GPT',
    GEMINI3_FLASH: 'Gemini'
}

def get_languages_list():
    languages_list = []
    seen_languages = set()

    for country_name in COUNTRIES_FILE:
        for lang, code in zip(COUNTRIES_FILE[country_name][LANGUAGES], COUNTRIES_FILE[country_name][LANGUAGES_CODE]):
            if lang not in seen_languages:
                languages_list.append({LANGUAGE: lang, LANGUAGES_CODE: code})
                seen_languages.add(lang)
    return languages_list
    