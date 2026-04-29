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
    if target_lang == "en":
        target_lang = "EN-GB"
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
        model_label = MODEL_LABEL[model_name]

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

def _build_question(
    base_text,
    scenario,
    country_name,
    language,
    language_code,
    model,
    model_exception=None,
):
    # NATIONALITY only → no translation
    if scenario == SCENARIO_NATIONALITY:
        return f"In {country_name}, {base_text}"

    # Build input text
    if scenario == SCENARIO_LANGUAGE:
        text = base_text
    else:  # LANGUAGE + NATIONALITY
        text = f"In {country_name}, {base_text}"

    # Translation logic
    if TRANSLATION_MODEL == DEEPL:
        if language == "Montenegrin" and model_exception:
            return model_translation(model_exception, text, language)
        return deepl_translation(model, text, language_code)

    return model_translation(model, text, language)

#translate the English Rainbow Meter questions prompt from to all the other languages and populate the scenario folders nested in the data/rainbow_meter folder            
def translate_rainbow_meter():
    model = Model(TRANSLATION_MODEL)
    if model.initialize_model():
        logger.error("translate_rainbow_meter: failed to initialize main model")
        return None

    model_exception = None
    if TRANSLATION_MODEL == DEEPL:
        model_exception = Model(TRANSLATION_MODEL_EXCEPTION)
        if model_exception.initialize_model():
            logger.error("translate_rainbow_meter: failed to initialize exception model")
            return None

    for scenario in SCENARIOS:
        result_path = f"{RAINBOW_METER_DATA_PATH}/{scenario}"

        for country_name, country_data in tqdm.tqdm(
            COUNTRIES_FILE.items(), desc=f"Translating ({scenario})"
        ):
            country_id = country_data[ID]
            languages = country_data[LANGUAGES]
            language_codes = country_data[LANGUAGES_CODE]

            for idx, language in enumerate(languages):
                language_code = language_codes[idx]

                # Initialize per-file data
                rainbow_meter = {
                    CATEGORY: [],
                    SUBCATEGORY: [],
                    **{q_type: [] for q_type in QUESTION_TYPES},
                }

                # Determine output path once
                if scenario == SCENARIO_LANGUAGE:
                    rm_path = f"{result_path}/rainbow_meter_{language_code}.csv"
                elif scenario == SCENARIO_NATIONALITY:
                    rm_path = f"{result_path}/rainbow_meter_{country_id}.csv"
                else:
                    rm_path = f"{result_path}/rainbow_meter_{language_code}_{country_id}.csv"

                if os.path.exists(rm_path):
                    #logger.info(f"Skipping existing file: {rm_path}")
                    continue

                for subcategory, row in RAINBOW_METER_EN.iterrows():
                    rainbow_meter[SUBCATEGORY].append(subcategory)
                    rainbow_meter[CATEGORY].append(row[CATEGORY])

                    for q_type in QUESTION_TYPES:
                        base_text = row[q_type].lower()

                        try:
                            question = _build_question(
                                base_text=base_text,
                                scenario=scenario,
                                country_name=country_name,
                                language=language,
                                language_code=language_code,
                                model=model,
                                model_exception=model_exception,
                            )
                        except Exception as e:
                            logger.error(f"translate_rainbow_meter: {e}")
                            question = None  # avoid undefined variable

                        rainbow_meter[q_type].append(question)

                # Save once per language/country
                df = pd.DataFrame(rainbow_meter)
                df.to_csv(rm_path, index=False, sep=";")

#translate the default prompt from English to all the other languages and populate the file prompt.json
def translate_default_prompt():
    model_name = DEEPL
    
    model = Model(model_name)
    error = model.initialize_model()
    if error:
        logging.error(f"translate_prompt")
    
    if TRANSLATION_MODEL == DEEPL:
        model_exception = Model(TRANSLATION_MODEL_EXCEPTION)
        error = model_exception.initialize_model()
        if error:
            logger.error(f"translate_rainbow_meter")
            return None
        
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
        
        #Iterate on every language and citizenship 
        for country_identity_num, language in enumerate(COUNTRIES_FILE[country_name][LANGUAGES]):
            language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][country_identity_num]
            
            if language not in row_results and language != "English":
                row_results[language] = {}

                prompt = row_results["English"]["prompt"]
                
                #for key, val in row_results["English"].items():
                if TRANSLATION_MODEL == DEEPL and language == "Montenegrin":
                    translation = model_translation(model_exception, prompt, language)
                elif TRANSLATION_MODEL == DEEPL:
                    translation = deepl_translation(model, prompt, language_code, "EN")
                else:
                    translation = model_translation(model, prompt, language)

                row_results[language]["prompt"] = translation
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
                    row_results[language][YES], row_results[language][NO] =  extracted[0], extracted[1]
                else:
                    logger.error(f"⚠️ Cannot detect the \"yes\" or \"no\" from the translated text in {language}: {translation}")
                    
            # Write ONCE after the loop
            with open("data/prompt.json", "w", encoding="utf-8") as f:
                json.dump(row_results, f, indent=4, ensure_ascii=False)
    

# #Check models ability to support the langauges bit back translation 
model_list = [QWEN35_27]
#test_systems_translation_abilities(model_list)

#Translate the prompt instructions
TRANSLATION_MODEL = DEEPL
# With TRANSLATION_MODEL = DEEPL this is necessary
TRANSLATION_MODEL_EXCEPTION = LLAMA32_3_OLL

translate_default_prompt()
#translate_rainbow_meter()


