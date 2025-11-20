from lib.constants import *
import json
import os

def get_country_list():
        countries_file = "data/countries.json"
        countries_file = open(countries_file)
        countries_file = json.load(countries_file)
        return countries_file

def get_country_language_list():
    country_list = get_country_list()
    
    for name in country_list:
        lan = country_list[name]['languages_code'][0]
        
    
class Country:
    def __init__(self, country_list, name):
        self.name = name
        self.country_id = country_list[name]["COUNTRY_ID"]
        self.language = country_list[name]['languages_code'][0]
    
    #Check if there is an existing file for that scenario, model and language
    def check_language_file(self, model_name):
        file_out = f"{RESULT_FOLDER+ SCENARIO_LANGUAGE_FOLDER+model_name}/{self.language}_raibow_meter.csv"
        return os.path.isfile(file_out)
    
    