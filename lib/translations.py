from constants import *
from models import *
import json
import requests
import random
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

# Calculate the similarity score using the embedding model "sentence-transformers/paraphrase-MiniLM-L6v2."
# Given two sentences, it return the similarity score (between 0 and 1)
def similarity_test(original, translated):
    try:
        response = requests.post(API_URL, headers = {
                "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
            }, json={
        "inputs": {
            "source_sentence": original,
            "sentences":[translated]
        }
    })
        if not(response.status_code == 200):
            logger.error(f"⚠️ Similarity Test: {response.reason}")
            return None
        response = response.json()[0]
        if response == None:
            breakpoint
        return response
    except Exception as X:
        logger.error(f"similarity_test: {X}")
        return None

#Translate the given text in the specified language using DeepL    
def deepl_translation(model, question, target_lang = "EN-US", source_lang = "EN"):
    if target_lang == "pt":
        target_lang = "pt-pt"
    if question == "" or target_lang == "cnr":
        return ""
    try:
        result = model.client.translate_text(question, target_lang=target_lang.upper(), source_lang=source_lang.upper())
        return result.text
    except Exception as X:
        logger.error(f"deepl_translation: {X}")
        return ""

#Translate the given text in the specified language using the given model
def model_translation(model, text, language = "English"):
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
            #From EN to the specified language 
            while translation == "" or translation == None:
                if model.model_name == DEEPL:
                    translation = deepl_translation(model, question, language[LANGUAGES_CODE])
                    if translation == "":
                        break
                else:
                    translation = model_translation(model, question, language[LANGUAGES])
            #From the specified language to EN
            while back_transated == "" or back_transated == None:
                if model.model_name == DEEPL:
                    back_transated = deepl_translation(model, translation, "EN-US", language[LANGUAGES_CODE])
                    if back_transated == "":
                        break
                else: 
                    back_transated = model_translation(model, translation)
        except Exception as X:
            logger.error(f"test_model: {X}")
            return None
        if model.model_name == DEEPL and translation == "" and "" == back_transated:
            similarity_score = 0
        else:
            similarity_score = similarity_test(question, back_transated) 
        similarity.append(similarity_score)
    return sum(similarity)/ len(similarity)

#Create the back_translation file, which contatins the average scores of back translation similarity test of every model in every language
def test_systems_translation_abilities(model_list):
    #Get the file
    result_path = f"data/translation_test/"
    os.makedirs(result_path, exist_ok=True)
    csv_path = f"{result_path}back_translation.csv"

    languages_list = []
    seen_languages = set()

    for country_name in COUNTRIES_FILE:
        for lang, code in zip(
            COUNTRIES_FILE[country_name][LANGUAGES],
            COUNTRIES_FILE[country_name][LANGUAGES_CODE],
        ):
            if lang not in seen_languages:
                languages_list.append({
                    LANGUAGES: lang,
                    LANGUAGES_CODE: code,
                })
                seen_languages.add(lang)

    # Load or create question list
    questions_path = f"{result_path}/questions.json"
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
            df.loc[row_idx, language[LANGUAGES]] = round(score, 2)

            # Compute the average score, without considering Montenegrin
            lang_scores = pd.to_numeric(df.loc[row_idx, [lang[LANGUAGES] for lang in languages_list if lang[LANGUAGES] != 'Montenegrin']], errors="coerce")
            df.loc[row_idx, "avg_score"] = round(lang_scores.mean(), 2)
            
            df.to_csv(csv_path, sep=";", index=False)

#translate the English Rainbow Meter questions prompt from to all the other languages and populate the scenario folders nested in the data/rainbow_meter folder            
def translate_rainbow_meter():
    result_path = f"{RAINBOW_METER_DATA_PATH}/{scenario}"

    model = Model(TRANSLATION_MODEL)
    error = model.initialize_model()
    if error:
        logger.error(f"translate_rainbow_meter")
        return None
    
    if TRANSLATION_MODEL == DEEPL:
        model_exception = Model(TRANSLATION_MODEL)
        error = model_exception.initialize_model()
        if error:
            logger.error(f"translate_rainbow_meter")
            return None
        
    #Iterate on the scenarios
    for scenario in SCENARIOS:
        
        rainbow_meter ={
            CATEGORY: [],
            SUBCATEGORY: []
        }
        
        for country_name, country_data in tqdm.tqdm(COUNTRIES_FILE.items(), desc=f"Translating prompt"):
            country_name = country_name
            country_id = country_data[ID]
            citizenship = country_data[CITIZENSHIP]
            
            #Iterate on every language and citizenship 
            for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
                language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]

            
                for type in QUESTION_TYPES:
                    rainbow_meter[type]= []

                #Iterate on the criteria
                for subcategory, row in RAINBOW_METER_EN.iterrows():
                    rainbow_meter[SUBCATEGORY].append(subcategory)
                    rainbow_meter[CATEGORY].append(row[CATEGORY])
                    for type in QUESTION_TYPES:
                        if scenario == SCENARIO_LANGUAGE: #SCENARIO LANGUAGE
                            rm_path = f"{result_path}/rainbow_meter_{language_code}.csv"
                            
                            try:
                                #Translate question from English in the country language
                                if TRANSLATION_MODEL == DEEPL and language == "Montenegrin":
                                    question = model_translation(model_exception, row[type].lower(), language)
                                elif TRANSLATION_MODEL == DEEPL:
                                    question = deepl_translation(model, row[type].lower(), language_code)
                                else:
                                    question = model_translation(model, row[type].lower(), language)  
                            except Exception as X:
                                logger.error(f"translate_rainbow_meter: {X}")
                                
                        elif scenario == SCENARIO_NATIONALITY: #SCENARIO NATIONALITY
                            rm_path = f"{result_path}/rainbow_meter_{country_id}.csv"
                            #Insert the country country in the question
                            question = f"In {country_name}, {row[type].lower()}"
                            
                        else: #SCENARIO LANGIAGE + NATIONALITY
                            rm_path = f"{result_path}/rainbow_meter_{language_code}_{country_id}.csv"
                            
                            #Insert the country in the questionand translate it from English in the country language
                            try:
                                if TRANSLATION_MODEL == DEEPL and language == "Montenegrin":
                                    question = model_translation(model_exception, row[type].lower(), language)
                                elif TRANSLATION_MODEL == DEEPL:
                                    question = deepl_translation(model, f"In {country_name}, {row[type].lower()}", language_code) 
                                else:
                                    question = model_translation(model, f"In {country_name}, {row[type].lower()}", language)
                            except Exception as X:
                                logger.error(f"translate_rainbow_meter: {X}")
                        
                        rainbow_meter[type].append(question)
                        
                    df = pd.DataFrame(rainbow_meter)
                    df.to_csv(rm_path, sep=";", index=False)

#translate the default prompt from English to all the other languages and populate the file prompt.json
def translate_default_prompt():
    model_name = DEEPL
    
    model = Model(model_name)
    error = model.initialize_model()
    if error:
        logging.error(f"translate_prompt")
        
    #Iterate on every country
    #for country_name, country_data in tqdm.tqdm(COUNTRIES_FILE.items(), total=len(COUNTRIES_FILE), desc=f"🔄 {self.model.model_name} - {self.scenario}"):
    row_results = {
        "English": {
            "prompt": 'Answer to the question with "yes" or "no" only.',
            YES: YES,
            NO: NO
        }
    }

    for country_name, country_data in tqdm.tqdm(COUNTRIES_FILE.items(), desc=f"Translating prompt"):
        country_name = country_name
        country_id = country_data[ID]
        citizenship = country_data[CITIZENSHIP]
        
        #Iterate on every language and citizenship 
        for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
            language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
            
            if language not in row_results and language != "English":
                row_results[language] = {}

                for key, val in row_results["English"].items():
                    if model.model_name == DEEPL:
                        translation = deepl_translation(model, val, language_code, "EN")
                    else:
                        translation = model_translation(model, val, language)

                    if key in [YES, NO]:
                        translation = (
                            translation.lower()
                            .replace(".", "")
                            .replace("*", "")
                            .replace('"', "")
                            .replace('\\"', "")
                            .strip()
                        )

                    row_results[language][key] = translation

            # Write ONCE after the loop
            with open("data/prompt.json", "w", encoding="utf-8") as f:
                json.dump(row_results, f, indent=4, ensure_ascii=False)
    

# #Check models ability to support the langauges bit back translation 
#model_list = [GPT54]
#test_systems_translation_abilities(model_list)

#Translate the prompt instructions
TRANSLATION_MODEL = DEEPL
TRANSLATION_MODEL_EXCEPTION = DEEPL
translate_default_prompt()

#translate_rainbow_meter()


