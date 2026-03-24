from constants import *
from models import *
from country import *
import json
import requests
import random
import deepl

logging.getLogger('deepl').setLevel(logging.WARNING)

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
    # text = re.sub(r"$begin:math:text$\[\^\)\]\{20\,\}$end:math:text$", "", text)

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

#Given two sentences, it return the similarity score (between 0 and 1)
def similarity_test(original, translated):
    payload = {
        "inputs": {
            "source_sentence": original,
            "sentences":[translated]
        }
    }
    try:
        response = requests.post(API_URL, headers = {
                "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
            }, json=payload)
        if not(response.status_code == 200):
            logger.error(f"⚠️ Similarity Test")
            return None
        response = response.json()[0]
        return response
    except Exception as X:
        logger.error(f"similarity_test: {X}")
        return None

#Translate the given text in the specified language using DeepL    
def deepl_translation(model, question, langauge_code = "EN-US"):
    if langauge_code == "pt":
        langauge_code = "pt-pt"
    if question == "":
        return ""
    try:
        result = model.client.translate_text(question, target_lang=langauge_code.upper())
        #if not(result.status_code == 200):
        #    logger.error(f"⚠️ deepl_translation")
        return result.text
    except Exception as X:
        return ""
        #logger.error(f"deepl_translation: {X}")

#Translate the given text in the specified language using the given model
def model_translation(model, text, language = "English"):
    # prompt = f"""Translate the following sentence in {language} {instruction}: 
    # {text}"""
    prompt = f"""Translate the text between <text> and </text> into {language}. Return ONLY the translation.
        <text>{text}</text>"""
    translation = ""
    try: 
        while translation == None or translation == "":
            translation = model.call_model(prompt)
            translation = clean_translation(translation)
        return translation
    except Exception as X:
        logger.error(f"translate: {X}")
        return None
        #return None

#Given a model, a question list and a language, it calculates the average similarity scores between the original questions from the question list and their translated version in the provided language
def test_system_translation(model, question_list, language):
    similarity = []
    for question in question_list:
        translation = ""
        back_transated = ""
        try: 
            while translation == "" or translation == None:
                if model.name == DEEPL:
                    translation = deepl_translation(model, question, language[LANGUAGES_CODE])
                    if translation == "":
                        break
                else:
                    translation = model_translation(model, question, language[LANGUAGES])
            while back_transated == "" or back_transated == None:
                if model.name == DEEPL:
                    back_transated = deepl_translation(model, translation)
                    if back_transated == "":
                        break
                else: 
                    back_transated = model_translation(model, translation)
        except Exception as X:
            logger.error(f"test_model: {X}")
            return None
        if model.name == DEEPL and translation == "" and "" == back_transated:
            similarity_score = 0
        else:
            similarity_score = similarity_test(question, back_transated) 
        similarity.append(similarity_score)
    return sum(similarity)/ len(similarity)

def test_systems_translation_abilities(model_list):
    result_path = f"{RESULT_PATH}/back_translation/"
    os.makedirs(result_path, exist_ok=True)
    csv_path = f"{result_path}bt_scores.csv"

    questions_path = "data/translation_test/questions.json"
    os.makedirs("data/translation_test", exist_ok=True)

    languages_list = [
        {
            LANGUAGES: COUNTRIES_FILE[country_name][LANGUAGES][0],
            LANGUAGES_CODE: COUNTRIES_FILE[country_name][LANGUAGES_CODE][0],
        }
        for country_name in COUNTRIES_FILE
    ]

    # Load or create question list
    if os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            questions_list = json.load(f)
    else:
        questions_list = []
        for type in QUESTION_TYPES:
            questions_list += random.sample(
                list(RAINBOW_METER_EN[type].values),
                NUM_SAMPLES_QUESTIONS
            )

        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(questions_list, f, ensure_ascii=False, indent=2)

    # Load or create dataframe
    columns = ["model"] + [lang[LANGUAGES] for lang in languages_list] + ["avg_score"]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")
    else:
        df = pd.DataFrame(columns=columns)

    # Ensure avg_score exists (in case file existed before)
    if "avg_score" not in df.columns:
        df["avg_score"] = None

    for model_name in model_list:

        model_label = MODELS_LABELS[model_name]

        # Ensure model row exists
        if model_label not in df["model"].values:
            new_row = {col: None for col in df.columns}
            new_row["model"] = model_label
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        row_idx = df.index[df["model"] == model_label][0]

        # --------- Skip model if all languages + avg_score are already filled ---------
        lang_values = df.loc[row_idx, [lang[LANGUAGES] for lang in languages_list]]
        avg_value = df.loc[row_idx, "avg_score"]
        if lang_values.notna().all() and pd.notna(avg_value):
            print(f"Skipping {model_label}: all languages already completed")
            continue

        model = Model(model_name)
        error = model.initialize_model()
        if error:
            continue

        for language in tqdm.tqdm(languages_list, desc=f"Testing {model_label}"):

            # Skip already computed languages
            if pd.notna(df.loc[row_idx, language[LANGUAGES]]):
                continue
            if language['languages_code'] == 'en':
                score = 1
            else:
                score = test_system_translation(model, questions_list, language)
            df.loc[row_idx, language[LANGUAGES]] = score

            # Recompute average score
            lang_scores = pd.to_numeric(df.loc[row_idx, [lang[LANGUAGES] for lang in languages_list]], errors="coerce")
            df.loc[row_idx, "avg_score"] = lang_scores.mean()

            # Save immediately
            df.to_csv(csv_path, sep=";", index=False)

#translate the English Rainbow Meter questions prompt from to all the other languages and populate the scenario folders nested in the data/rainbow_meter folder            
def translate_rainbow_meter():
    model = GEMMA3_27
    #Iterate on the countries
    for country_name in tqdm.tqdm(COUNTRIES_FILE, desc="Generating Rainbow Meter Questions", total=len(COUNTRIES_FILE)): 
        country = Country(country_name)
        
        #Iterate on the scenarios
        for scenario in SCENARIOS:
            result_path = f"data/{RAINBOW_METER_PATH}/{scenario}"
            if scenario == SCENARIO_LANGUAGE:
                rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{country.language_code}.csv"
            elif scenario == SCENARIO_NATIONALITY:
                rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{country.id}.csv"
            else:
                rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{country.language_code}_{country.id}.csv"
                
            rainbow_meter ={
                CATEGORY: [],
                SUBCATEGORY: []
            }
            for type in QUESTION_TYPES:
                rainbow_meter[type]= []

            for subcategory, row in RAINBOW_METER_EN.iterrows():
                rainbow_meter[SUBCATEGORY].append(subcategory)
                rainbow_meter[CATEGORY].append(row[CATEGORY])
                for type in QUESTION_TYPES:
                    if scenario == SCENARIO_LANGUAGE:
                        #Translate question from English in the country language
                        try:
                            question = model_translation(model, row[type].lower(), country.language)  
                        except Exception as X:
                            logger.error(f"translate_rainbow_meter: {X}")
                    elif scenario == SCENARIO_NATIONALITY:
                        #Insert the country country in the question
                        question = f"In {country.name}, {row[type].lower()}"
                    else:
                        #Insert the country in the questionand translate it from English in the country language
                        try: 
                            question = model_translation(model, f"In {country.name}, {row[type].lower()}", country.language)
                        except Exception as X:
                            logger.error(f"translate_rainbow_meter: {X}")
                    
                    rainbow_meter[type].append(question)
                    
                os.makedirs(f"data/{RAINBOW_METER_PATH}/{scenario}", exist_ok=True)
                df = pd.DataFrame(rainbow_meter)
                df.to_csv(rm_path, sep=";", index=False)

#translate the default prompt from English to all the other languages and populate the file prompt.json
def translate_default_prompt():
    model_name = DEEPL
    
    # languages_list = [COUNTRIES_FILE[country_name][LANGUAGES][0] for country_name in COUNTRIES_FILE]
    languages_list = [
        {
            LANGUAGES: COUNTRIES_FILE[country_name][LANGUAGES][0],
            LANGUAGES_CODE: COUNTRIES_FILE[country_name][LANGUAGES_CODE][0],
        }
        for country_name in COUNTRIES_FILE
    ]
    model = Model(model_name)
    error = model.initialize_model()
    if error:
        logging.error(f"translate_prompt")
    
    row_results = {
        "English":{
            "prompt":'Answer to the question with "yes" or "no" only.',
            YES: "yes", 
            NO: "no"
            }
        }
    
    for language in tqdm.tqdm(languages_list, desc=f"Translating prompt"):
        if language[LANGUAGES] not in row_results and language[LANGUAGES] != "English":
            row_results[language[LANGUAGES]] = {}
            for key, val in row_results["English"].items():
                if model.name == DEEPL:
                    translation = deepl_translation(model, val, language[LANGUAGES_CODE])
                    if translation == "":
                        breakpoint
                else: 
                    translation = model_translation(model, val, language[LANGUAGES])
                row_results[language[LANGUAGES]][key] = translation.lower().replace(".", "").replace("*", "").replace('"', "").replace('\\"', "").strip() if key == YES or key == NO else translation 
    
    with open("data/prompt.json", "w", encoding="utf-8") as f:
        json.dump(row_results, f, indent=4, ensure_ascii=False)
    

#Check models ability to support the langauges
model_list = [GPT5]
test_systems_translation_abilities(model_list)

#Translate the prompt instructions
#translate_default_prompt()

#translate_rainbow_meter()


