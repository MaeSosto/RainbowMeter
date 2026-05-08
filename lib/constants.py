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
TABLES_PATH = f"tables"
os.makedirs(TABLES_PATH, exist_ok=True)
RAINBOW_METER_DATA_PATH = "data/rainbow_meter"


SCENARIO_LANGUAGE = "language_scenario"
SCENARIO_NATIONALITY = "nationality_scenario"
SCENARIO_LAN_NAT = "language_nationality_scenario"
SCENARIOS = [SCENARIO_LANGUAGE, SCENARIO_NATIONALITY, SCENARIO_LAN_NAT]
MODELS_STATS_PATH = "models_stats"
for s in SCENARIOS:
    os.makedirs(f"{RAINBOW_METER_RESULT_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{EVALUATIONS_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{GRAPHS_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{GRAPHS_PATH}/{MODELS_STATS_PATH}/{s}", exist_ok=True)
    os.makedirs(f"{TABLES_PATH}/{MODELS_STATS_PATH}/{s}", exist_ok=True)
    
RAINBOW_METER_EN = pd.read_csv(f"{RAINBOW_METER_DATA_PATH}/{SCENARIO_LANGUAGE}/rainbow_meter_en.csv", sep=";", index_col=SUBCATEGORY)
