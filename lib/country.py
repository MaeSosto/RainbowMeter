from lib.constants import *
import json
import os

folder_result = "results_for_analysis/"
folder_language_scenario = "language_scenario/"

def get_country_list():
        countries_file = "data/countries.json"
        countries_file = open(countries_file)
        countries_file = json.load(countries_file)
        return countries_file
        
class Country:
    def __init__(self, country_list, name):
        self.name = name
        self.country_id = country_list[name]["COUNTRY_ID"]
        self.language = country_list[name]['languages_code'][0]
    
    #Check if there is a 
    def check_language_file(self, model_name):
        path_result = folder_result+ folder_language_scenario+model_name 
        file_out = f"{path_result}/{self.language}_raibow_meter.csv"
        return os.path.isfile(file_out)
    
    