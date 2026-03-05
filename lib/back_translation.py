from constants import *
from models import *
from country import *
import json
import requests
import random

HF_TOKEN = os.getenv('HF_TOKEN')
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

NUM_SAMPLES_QUESTIONS = 3

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
    
        try: 
            transation = translate(model, question, language)
            back_transated = translate(model, transation)
            similarity.append(similarity_test(question, back_transated) )
        except Exception as X:
            breakpoint
            logger.error(f"test_model: {X}")
            return None
    return sum(similarity)/ len(similarity)

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

def test_model_languages(model):
    #List of all the languages
    languages_list = [COUNTRIES_FILE[country_name][LANGUAGES][0] for country_name in COUNTRIES_FILE]
    #languages_list = languages_list[:3]
    #languages_code = [COUNTRIES_FILE[country_name][LANGUAGES_CODE][0] for country_name in COUNTRIES_FILE]

    questions_list = []
    for type in QUESTION_TYPES:
        questions_list = questions_list + random.sample(list(RAINBOW_METER_EN[type].values), NUM_SAMPLES_QUESTIONS)
        
    results = {
        "model": []
    }
    for language in languages_list:
        results[language] = []
    
    results["model"].append(model.name)
    for language in tqdm.tqdm(languages_list, total=len(languages_list), desc=f"Testing {model.name}"):
        #logger.info(f"{model} - {language}")
        if len(results[language]) == 0 : #Language evaluated already
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
    #logger.info(f"Testing languages with {model.name}")
    test_model_languages(model)


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




## [0.605, 0.894]
