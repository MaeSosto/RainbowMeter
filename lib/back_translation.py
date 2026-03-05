from constants import *
from models import *
from country import *
import json
import requests
import random

HF_TOKEN = os.getenv('HF_TOKEN')
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

NUM_SAMPLES_QUESTIONS = 3
instruction = "without any additional text or note"

def similarity_test(original, translated):
    payload = {
        "inputs": {
            "source_sentence": original,
            "sentences":[translated]
        }
    }
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"}, json=payload)
    return response.json()[0]

def test_model(model, question_list, language):
    
    similarity = []
    #print(questions_list)
    for question in question_list:
        prompt = f"""Translate the following sentence in {language}
        {instruction}:{question}"""
        try: 
            transation = model.call_model(prompt)
            prompt = f"""Translate the following sentence in english {instruction}: {transation}"""
            back_transated = model.call_model(prompt)
            similarity.append(similarity_test(question, back_transated))
        except Exception as X:
            logger.error(X)
            return None
    return sum(similarity)/ len(similarity)

def translate(model, text, language):
    prompt = f"""Translate the following sentence in {language}
{instruction}: {text}"""
    try: 
        transation = model.call_model(prompt)
        return transation
    except Exception as X:
        logger.error(X)
        return None
        
def translate_rainbow_meter(model, language, language_code):
    result_path = f"data/{RAINBOW_METER_PATH}/{SCENARIO_LANGUAGE}/"
    rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{language_code}.csv"
    translated_rm ={
        CATEGORY: [],
        SUBCATEGORY: []
    }
    for type in QUESTION_TYPES:
        translated_rm[type]= []

    for subcategory, row in RAINBOW_METER_EN.iterrows():
        translated_rm[SUBCATEGORY].append(subcategory)
        translated_rm[CATEGORY].append(row[CATEGORY])
        for type in QUESTION_TYPES:
            translated_rm[type].append(translate(model, row[type], language))
        
        os.makedirs(result_path, exist_ok=True)
        df = pd.DataFrame(translated_rm)
        df.to_csv(rm_path, sep=";", index=False)
# sentences = ["I'm very happy", "I'm filled with happiness"]
# print(similarity_test(sentences))

def test_model_languages(model):
    #List of all the languages
    languages_list = [COUNTRIES_FILE[country_name][LANGUAGES][0] for country_name in COUNTRIES_FILE]
    #languages_list = languages_list[:3]
    #languages_code = [COUNTRIES_FILE[country_name][LANGUAGES_CODE][0] for country_name in COUNTRIES_FILE]

    questions_list = []
    for type in QUESTION_TYPES:
        questions_list = questions_list + random.sample(list(RAINBOW_METER_EN[type].values), NUM_SAMPLES_QUESTIONS)
        
    results = {
        "model": model_name
    }
    for language in languages_list:
        results[language] = []
    
    for language in tqdm.tqdm(languages_list, total=len(languages_list), desc=f"Testing {model.name} in {language}"):
        #logger.info(f"{model} - {language}")
        results[language].append(test_model(model, questions_list, language))
        
    result_path = f"{RESULT_PATH}/back_translation/"
    os.makedirs(result_path, exist_ok=True)
    df = pd.DataFrame(results)
    #print(df)
    df.to_csv(f"{result_path}{model.name}_bt_scores.csv", sep=";", index=False)


model_name = LLAMA3
model = Model(model_name)
error = model.initialize_model()
if not error:
    logger.info(f"Testing languages with {model}")
    test_model_languages(model_name)


#check if the model support the language
    
# for country_name in COUNTRIES_FILE: 
#     id = COUNTRIES_FILE[country_name][ID]
#     language = COUNTRIES_FILE[country_name][LANGUAGES][0]
#     language_code = COUNTRIES_FILE[country_name][LANGUAGES_CODE][0]
#     citizenship = COUNTRIES_FILE[country_name][CITIZENSHIP]

#     result_path = f"{RAINBOW_METER_PATH}/"
#     rm_path = f"{result_path}/{RAINBOW_METER_PATH}_{language_code}.csv"
    
#     if not os.path.exists(rm_path):
#         translate_rainbow_meter(model, language, language_code)




## [0.605, 0.894]
