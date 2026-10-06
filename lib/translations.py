from constants import *
from models import *
import json
import requests
import random
logging.getLogger('deepl').setLevel(logging.WARNING)

RAINBOW_METER_EN = pd.read_csv(f"{RAINBOW_METER_DATA_PATH}/{SCENARIO_LANGUAGE}/rainbow_meter_en.csv", sep=";", index_col=CRITERION_ID)
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
NUM_SAMPLES_QUESTIONS = 3

#Clean the given text
def clean_translation(text):
    if text is None:
        return ""
    text = text.strip()
    # remove everything starting with "Note:"
    text = re.split(r'\bNote:\b', text, flags=re.IGNORECASE)[0]
    # remove markdown emphasis
    text = re.sub(r"[*_`]+", "", text)
    # remove leading labels like "Translation:", "Në shqip:", etc.
    text = re.sub(r"^[A-Za-zÀ-ÖØ-öø-ÿ\s]+:\s*", "", text)
    # # remove long parenthetical commentary
    text = re.sub(r"$begin:math:text$\[\^\)\]\{20\,\}$end:math:text$", "", text)
    # split into lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    # remove obvious explanation lines
    filtered = []
    for l in lines:
        if re.search(r"(translation|explanation|comment)", l, re.I):
            continue
        filtered.append(l)
    if not filtered:
        filtered = lines
    # choose the longest line (usually the translation)
    return max(filtered, key=len)

#translate the English Rainbow Meter questions prompt from to all the other languages and populate the scenario folders nested in the data/rainbow_meter folder            
def translate_rainbow_meter():
    if MAIN_TRANSLATOR_MODEL == DEEPL and MODEL_MONTENEGRIN_EXCEPTION != "":
        model_montenegrin = Model(MODEL_MONTENEGRIN_EXCEPTION)
    if model_montenegrin.initialize_model():
        logger.error("translate_rainbow_meter: failed to initialize montenegrin model")
        return None
    
    model = Model(MAIN_TRANSLATOR_MODEL)
    if model.initialize_model():
        logger.error("translate_rainbow_meter: failed to initialize main translation model")
        return None

    err = False
    #Iterate on scenarios
    for scenario in SCENARIOS:
        result_path = f"{RAINBOW_METER_DATA_PATH}/{scenario}"
        
        #Iterate on countries
        for country_name, country_data in tqdm.tqdm(COUNTRIES_FILE.items(), desc=f"Translating ({scenario})"):
            country_id = country_data[ID]
            languages = country_data[LANGUAGES]
            language_codes = country_data[LANGUAGES_CODE]
            
            #Iterate on languages of that country
            for idx, language in enumerate(languages):
                language_code = language_codes[idx]

                # Initialize per-file data
                rainbow_meter = {
                    CRITERION_ID: [],
                    **{q_type: [] for q_type in QUESTION_TYPES},
                }

                # Determine output path once
                if scenario == SCENARIO_LANGUAGE:
                    rm_path = f"{result_path}/rainbow_meter_{language_code}.csv"
                elif scenario == SCENARIO_COUNTRY:
                    rm_path = f"{result_path}/rainbow_meter_{country_id}.csv"
                else:
                    rm_path = f"{result_path}/rainbow_meter_{language_code}_{country_id}.csv"
                
                if os.path.exists(rm_path):
                    #logger.info(f"Skipping existing file: {rm_path}")
                    continue
                
                
                for idx, row in RAINBOW_METER_EN.iterrows():
                    rainbow_meter[CRITERION_ID].append(idx)

                    for q_type in QUESTION_TYPES:
                        base_text = row[q_type].lower()

                        try:
                            # Country only → no translation
                            if scenario == SCENARIO_COUNTRY or language == "English":
                                question = f"In {"the " if country_name == "United Kingdom" or country_name == "Netherlands" else ""}{country_name}, {base_text}"
                            else:
                                # Build input text
                                text = base_text if scenario == SCENARIO_LANGUAGE else f"In {country_name}, {base_text}"  # LANGUAGE + Country

                                # Translation logic
                                if (language == "Montenegrin" or language_codes == "cnr") and MAIN_TRANSLATOR_MODEL == DEEPL and MODEL_MONTENEGRIN_EXCEPTION != "":
                                    question = clean_translation(model_translation(model_montenegrin, text, language))
                                else:
                                    question = deepl_translation(model, text, language_code) if MAIN_TRANSLATOR_MODEL == DEEPL else model_translation(model, text, language) 
                                
                        except Exception as e:
                            logger.error(f"translate_rainbow_meter: {e}")
                            err = True
                            break

                        rainbow_meter[q_type].append(question)
                        print(rainbow_meter)
                    # Save once per language/country
                    df = pd.DataFrame(rainbow_meter)
                    df.to_csv(rm_path, index=False, sep=";")
            if err:
                err = False
                break

#translate the default prompt from English to all the other languages and populate the file prompt.json
def translate_prompt():
    if MAIN_TRANSLATOR_MODEL == DEEPL and MODEL_MONTENEGRIN_EXCEPTION != "":
        model_montenegrin = Model(MODEL_MONTENEGRIN_EXCEPTION)
    if model_montenegrin.initialize_model():
        logger.error("translate_prompt: failed to initialize montenegrin model")
        return None
    
    model = Model(MAIN_TRANSLATOR_MODEL)
    if model.initialize_model():
        logger.error("translate_prompt: failed to initialize main translation model")
        return None
        

    #Get the already translated prompts
    prompt_path = f"data/prompt.json"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_json = json.load(f)
    else:
        prompt_json = {
            "English": {
                "prompt": 'Answer to the question with "yes" or "no" only.',
                YES: YES,
                NO: NO
            }
        }
        
    #Iterate on countries
    for country_name, country_data in tqdm.tqdm(COUNTRIES_FILE.items(), desc=f"Translating prompts"):
        languages = country_data[LANGUAGES]
        language_codes = country_data[LANGUAGES_CODE]
        
        #Iterate on languages of that country
        for idx, language in enumerate(languages):
            language_code = language_codes[idx]
            
            if language not in prompt_json and language != "English":
                prompt_json[language] = {}
                prompt = prompt_json["English"]["prompt"]
                
                #for key, val in row_results["English"].items():
                # if translation_model == DEEPL and language == "Montenegrin":
                #     translation = model_translation(model_exception, prompt, language)
                # el
                # if MAIN_TRANSLATOR_MODEL == DEEPL:
                #     translation = deepl_translation(model, prompt, language_code, "EN")
                # else:
                #     translation = model_translation(model, prompt, language)
                
                    # Translation logic
                if (language == "Montenegrin" or language_code == "cnr") and MAIN_TRANSLATOR_MODEL == DEEPL and MODEL_MONTENEGRIN_EXCEPTION != "":
                    translation = model_translation(model_montenegrin, prompt, language)
                else:
                    translation = deepl_translation(model, prompt, language) if MAIN_TRANSLATOR_MODEL == DEEPL else model_translation(model, prompt, language) 
                

                prompt_json[language]["prompt"] = translation
                pattern = r'''
                        "([^"]+)"          |   # double quotes
                        «([^»]+)»         |   # guillemets
                        “([^”]+)”         |   # curly double
                        ‘([^’]+)’         |   # curly single
                        '([^']+)'             # straight single
                    '''
                matches = re.findall(pattern, translation, re.VERBOSE)
                extracted = []
                for match in matches:
                    # each match is a tuple, only one group filled
                    for group in match:
                        if group:
                            extracted.append(group.strip())

                if len(extracted) >= 2:
                    prompt_json[language][YES], prompt_json[language][NO] =  extracted[0], extracted[1]
                else:
                    logger.error(f"⚠️ Cannot detect the \"yes\" or \"no\" from the translated text in {language}: {translation}")
                    
                # Write ONCE after the loop
                with open("data/prompt.json", "w", encoding="utf-8") as f:
                    json.dump(prompt_json, f, indent=4, ensure_ascii=False)
    
#Translate the prompt instructions
MAIN_TRANSLATOR_MODEL = DEEPL
# With TRANSLATION_MODEL = DEEPL this is necessary
MODEL_MONTENEGRIN_EXCEPTION = QWEN35_27

translate_prompt()
translate_rainbow_meter()
