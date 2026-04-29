from lib.models import *
from lib.constants import *

#Given a lan id, return the language name
def get_lang_from_lang_code(lang_code = "", count_code = ""):
    for country_name in COUNTRIES_FILE: 
            for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                if lang_code == COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num] and (count_code == "" or count_code == COUNTRIES_FILE[country_name][ID]):
                    return language, country_name
                if count_code == COUNTRIES_FILE[country_name][ID] and lang_code == "":
                    return language, country_name
    return None

for scenario in SCENARIOS:
    root_dir = f"{RAINBOW_METER_RESULT_PATH}/{scenario}"
    for model_name in MODEL_LIST:
        model_path = os.path.join(root_dir, model_name)
        model_label = MODEL_LABEL[model_name]
        if not os.path.exists(model_path):
            continue
        
        for file in os.listdir(model_path):
            # extract language (e.g., az from rm_answers_az.csv)
            
            label = file.replace("rm_answers_", "").replace(".csv", "")
            if scenario == SCENARIO_LANGUAGE:
                language, country_name = get_lang_from_lang_code(label)
                label = language
            elif scenario == SCENARIO_NATIONALITY:
                language, country_name = get_lang_from_lang_code("", label)
                label = country_name
            elif scenario == SCENARIO_LAN_NAT:
                language, country_name = get_lang_from_lang_code(label.split("_")[0], label.split("_")[1])
                label = f"{language} - {country_name}"
            
            if language in ["German","Azerbaijani", "Dutch", "Hungarian", "Icelandic", "Irish", "Latvian", "Luxembourgish"]:
                file_path = os.path.join(model_path, file) 
                #print(file_path)
                os.remove(file_path)