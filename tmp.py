
from lib.constants import *

QWEN35_2 = "Qwen/Qwen3.5-2B" #HF
QWEN35_9 = "Qwen/Qwen3.5-9B" #HF
QWEN35_27 = "Qwen/Qwen3.5-27B" #HF
LLAMA32_3 = "meta-llama/Llama-3.2-3B" #HF
LLAMA32_3_OLL ="llama3.2:3b" #Ollama
LLAMA31_8 = "meta-llama/Llama-3.1-8B" #HF
LLAMA31_8_OLL = "llama3.1:8b" #Ollama
LLAMA31_70 = "meta-llama/Llama-3.1-70B" #HF
LLAMA31_70_OLL = "llama3.1:70b" #Ollama
DEEPSEEKV32 = "deepseek-chat"
SONNET46 = "claude-sonnet-4-6"
GPT54 = 'gpt-5.4'
GPT54_MINI = 'gpt-5.4-mini'
GEMINI3_FLASH = 'gemini-3-flash-preview'
DEEPL = "DeepL" 

MODEL_LIST = [QWEN35_2, QWEN35_9, QWEN35_27, LLAMA32_3_OLL, LLAMA31_8_OLL, LLAMA31_70_OLL, DEEPSEEKV32, SONNET46, GPT54, GEMINI3_FLASH]

MODEL_LABEL = {
    QWEN35_2: "Qwen3.5 2B",
    QWEN35_9: "Qwen3.5 9B",
    QWEN35_27: "Qwen3.5 27B",
    LLAMA32_3: "LlaMa 3.2 3B",
    LLAMA32_3_OLL: "LlaMa 3.2 3B",
    LLAMA31_8_OLL: "LlaMa 3.1 8B",
    LLAMA31_8: "LlaMa 3.1 8B",
    LLAMA31_70: "LlaMa 3.1 70B",
    LLAMA31_70_OLL: "LlaMa 3.1 70B",
    DEEPSEEKV32: "DeepSeek-V3.2",
    SONNET46: "Sonnet 4.6",
    DEEPL: "DeepL",
    GPT54 : 'GPT 5.4',
    GPT54_MINI : 'GPT 5.4-mini',
    GEMINI3_FLASH: 'Gemini 3 Flash'
}

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
                
                if language in ["Luxembourgish", "Montenegrin"]:
                    file_path = os.path.join(model_path, file) 
                    #print(file_path)
                    os.remove(file_path)
            elif scenario == SCENARIO_NATIONALITY:
                language, country_name = get_lang_from_lang_code("", label)
                label = country_name
            elif scenario == SCENARIO_LAN_NAT:
                language, country_name = get_lang_from_lang_code(label.split("_")[0], label.split("_")[1])
                label = f"{language} - {country_name}"
            
                if language in ["Luxembourgish", "Montenegrin"]:
                    file_path = os.path.join(model_path, file) 
                    #print(file_path)
                    os.remove(file_path)