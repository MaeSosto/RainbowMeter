import pandas as pd
import json
from tqdm import tqdm
import csv
import json

CRITERIA_PATH = 'data\criteria.csv' 

LANGUAGES = ['EN']

def _convertCSVtoJSON(file_in, file_out):
    with open(file_in, mode='r', newline='', encoding='utf-8') as csvfile:
        data = list(csv.DictReader(csvfile))

    with open(file_out, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4)