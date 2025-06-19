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
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

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
LLAMA3_70B = 'llama3:70b'
GEMMA3 = 'gemma3'
GEMMA3_27B = 'gemma3:27b'
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'


def _convertCSVtoJSON(file_in, file_out):
    with open(file_in, mode='r', newline='', encoding='utf-8') as csvfile:
        data = list(csv.DictReader(csvfile))

    with open(file_out, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4)