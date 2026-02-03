import json
import csv
import logging
import torch
from tqdm import tqdm
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

torch.set_default_device(device)
logger.info(f"Using device: {device}")

LLAMA3 = 'llama3'
LLAMA4 = 'llama4'
LLAMA3_70B = 'llama3:70b'
GEMMA3 = 'gemma3'
GEMMA3_27B = 'gemma3:27b'
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
DEEPSEEK = 'deepseek-r1'
DEEPSEEK_671B = 'deepseek-reasoner'
MODEL_LIST = [LLAMA3, LLAMA3_70B, LLAMA4, GEMMA3, GEMMA3_27B, GPT4, GPT4_MINI, GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_LITE, DEEPSEEK, DEEPSEEK_671B]

MODELS_LABELS = {
    LLAMA3 : 'Llama 3',
    LLAMA3_70B : 'Llama 3(70b)',
    LLAMA4 : 'Llama 4',
    GEMMA3 : 'Gemma 3',
    GEMMA3_27B : 'Gemma 3(27b)',
    GEMINI_2_0_FLASH : "Gemini 2.0 Flash",
    GEMINI_2_0_FLASH_LITE : "Gemini 2.0 Flash Lite",
    GPT4_MINI : 'GPT4o Mini',
    GPT4 : 'GPT4o',
    DEEPSEEK: 'DeepSeek R1',
    DEEPSEEK_671B: 'DeepSeek R1 (671b)',
}

QUESTION_SUPPORT = "Support"
QUESTION_OPPOSITION = "Opposition"
QUESTION_FACT = "Fact"
PROMPT_TYPES = [QUESTION_FACT, QUESTION_SUPPORT, QUESTION_OPPOSITION] 
RESULT_FOLDER = "results/"
SCENARIO_LANGUAGE_FOLDER = "language_scenario/"
RAINBOW_METER_DATA_PATH = "data/rainbow_meter/"
RAINBOW_MAP_DATA_PATH = "data/rainbow_map/"
CRITERIA_FILE = "rainbow_map/criteria.json"
COUNTRIES_FILE = "data/countries.json"
RAINBOW_MAP_DATA_PATH = "data/rainbow_map/"
CRITERIA_FILE = "rainbow_map/criteria.json"
COUNTRIES_FILE = "data/countries.json"
GRAPHS_FOLDER = f'graphs/'


        
def csv_to_json(path_json, path_csv):
    csv_file = pd.DataFrame(pd.read_csv(path_csv, sep = ",", header = 0, index_col = False))
    csv_file = pd.DataFrame(pd.read_csv(path_csv, sep = ",", header = 0, index_col = False))
    csv_file.to_json(path_json, indent = 4, orient = "records", double_precision = 10, force_ascii = True, date_unit = "ms", default_handler = None)
    
def json_to_csv(path_json, path_csv):
    df = pd.read_json(path_json)
    df.to_csv(path_csv)
    
    
csv_to_json("data/rainbow_meter/rainbow_meter_en_.json", "data/rainbow_meter/rainbow_meter_en.csv")
