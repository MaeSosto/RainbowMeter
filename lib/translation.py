from constants import *
from models import *
from country import *
import json
import requests
import random


HF_TOKEN = os.getenv('HF_TOKEN')
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

NUM_SAMPLES_QUESTIONS = 1

#Given two sentences, it return the similarity score (between 0 and 1)
def similarity_test(original, translated):
    payload = {
        "inputs": {
            "source_sentence": original,
            "sentences":[translated]
        }
    }
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"}, json=payload)
    return response.json()[0]

#Given a model, a question list and a language, it calculates the average similarity scores between the original questions from the question list and their translated version in the provided language
def test_model(model, question_list, language):
    similarity = []
    for question in question_list:
    
        try: 
            transation = translate(model, question, language)
            back_transated = translate(model, transation)
            similarity.append(similarity_test(question, back_transated) )
        except Exception as X:
            breakpoint
            logger.error(f"test_model: {X}")
            return None
    return sum(similarity)/ len(similarity)

#Translate the given text in the specified language using the given model
def translate(model, text, language = "English"):
    # prompt = f"""Translate the following sentence in {language} {instruction}: 
    # {text}"""
    prompt = f"""Translate the text between <text> and </text> into {language}.
        Return ONLY the translation.

        <text>
        {text}
        </text>"""
    try: 
        transation = model.call_model(prompt)
        if transation == None:
            return ""
        #     logger.error(f"translate: {X}")
        #     return None
        return transation.strip().split("\n")[0]
    except Exception as X:
        logger.error(f"translate: {X}")
        return None
        
def translate_rainbow_meter(model, scenario, country, rm_path):
        
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
                    question = translate(model, row[type].lower(), country.language)  
                except Exception as X:
                    logger.error(f"translate_rainbow_meter: {X}")
            elif scenario == SCENARIO_NATIONALITY:
                #Insert the country country in the question
                question = f"In {country.name}, {row[type].lower()}"
            else:
                #Insert the country in the questionand translate it from English in the country language
                try: 
                    question = translate(model, f"In {country.name}, {row[type].lower()}", country.language)
                except Exception as X:
                    logger.error(f"translate_rainbow_meter: {X}")
            
            rainbow_meter[type].append(question)
            
        result_path = f"data/{RAINBOW_METER_PATH}/{scenario}"
        os.makedirs(result_path, exist_ok=True)
        df = pd.DataFrame(rainbow_meter)
        df.to_csv(rm_path, sep=";", index=False)
# sentences = ["I'm very happy", "I'm filled with happiness"]
# print(similarity_test(sentences))


def test_model_languages(model_list):
    result_path = f"{RESULT_PATH}/back_translation/"
    os.makedirs(result_path, exist_ok=True)
    csv_path = f"{result_path}bt_scores.csv"

    languages_list = [COUNTRIES_FILE[country_name][LANGUAGES][0] for country_name in COUNTRIES_FILE]

    # Load or create dataframe
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")
    else:
        df = pd.DataFrame(columns=["model"] + languages_list)

    # Get random questions
    questions_list = []
    for type in QUESTION_TYPES:
        questions_list += random.sample(list(RAINBOW_METER_EN[type].values), NUM_SAMPLES_QUESTIONS)

    for model_name in model_list:

        model_label = MODELS_LABELS[model_name]

        # Ensure model row exists
        if model_label not in df["model"].values:
            new_row = {col: None for col in df.columns}
            new_row["model"] = model_label
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        row_idx = df.index[df["model"] == model_label][0]

        model = Model(model_name)
        error = model.initialize_model()
        if error:
            continue

        for language in tqdm.tqdm(languages_list, desc=f"Testing {model_label}"):

            # Skip already computed languages
            if pd.notna(df.loc[row_idx, language]):
                continue

            score = test_model(model, questions_list, language)

            df.loc[row_idx, language] = score

            # Save immediately after each language
            df.to_csv(csv_path, sep=";", index=False)

#translate the default prompt from English to all the other languages and populate the file prompt.json
def translate_default_prompt():
    model_name = GEMMA3_4
    
    languages_list = [COUNTRIES_FILE[country_name][LANGUAGES][0] for country_name in COUNTRIES_FILE]
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
        if language not in row_results and language != "English":
            row_results[language] = {}
            for key, val in row_results["English"].items():
                tra = translate(model, val, language)
                row_results[language][key] = tra.lower().replace(".", "").replace("*", "").replace('"', "").replace('\\"', "").strip() if key == YES or key == NO else tra 
    
    with open("data/prompt.json", "w", encoding="utf-8") as f:
        json.dump(row_results, f, indent=4, ensure_ascii=False)
    
model_list = [QWEN3_4, QWEN3_30, GEMMA3_4, GEMMA3_12, GEMMA3_27]

test_model_languages(model_list)
#translate_prompt()

    #check if the model support the language
        
        # for country_name in tqdm.tqdm(COUNTRIES_FILE, desc="Generating Rainbow Meter Questions", total=len(COUNTRIES_FILE)): 
        #     country = Country(country_name)

        #     for scenario in SCENARIOS:
        #         result_path = f"data/{RAINBOW_METER_PATH}/{scenario}"
        #         if scenario == SCENARIO_LANGUAGE:
        #             rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{country.language_code}.csv"
        #         elif scenario == SCENARIO_NATIONALITY:
        #             rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{country.id}.csv"
        #         else:
        #             rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{country.language_code}_{country.id}.csv"
                
        #         if not os.path.exists(rm_path):
        #             translate_rainbow_meter(model, scenario, country, rm_path)


