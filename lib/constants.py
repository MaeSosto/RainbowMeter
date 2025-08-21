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


def csv_to_json(csv_path, json_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        data = list(csv.DictReader(csvfile))

    with open(json_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4)
        
def json_to_csv(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)