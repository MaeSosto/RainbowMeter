from constants import *
from models import *
from langdetect import detect
from langdetect import DetectorFactory
from lingua import Language, LanguageDetectorBuilder
import random

DetectorFactory.seed = 0
SIMILARITY_API = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
NUM_SAMPLES_QUESTIONS = 3

api_key = os.getenv('DEEPSEEK_API_KEY')
if api_key is None:
    logger.error(f"⚠️ DEEPSEEK_API_KEY is missing")
else:
    DEEPL_TRANSLATOR = deepl.DeepLClient("your-api-key")
    
def detect_language(text):
    result = DEEPL_TRANSLATOR.translate_text(
        text,
        target_lang="EN-GB"
    )
    return result.detected_source_lang

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

def load_create_question_list(languages_list):
    # Load or create question list
    questions_path = f"data/questions.json"
    if os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            questions_list = json.load(f)
    else:
        questions_list = []
        for type in QUESTION_TYPES:
            questions_list += random.sample(list(RAINBOW_METER_EN[type].values), NUM_SAMPLES_QUESTIONS)

        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(questions_list, f, ensure_ascii=False, indent=2)

    # Load or create dataframe
    return ["model"] + [lang[LANGUAGES] for lang in languages_list] + ["avg_score"], questions_list
    
# Calculate the similarity score using the embedding model "sentence-transformers/paraphrase-MiniLM-L6v2."
# Given two sentences, it return the similarity score (between 0 and 1)
def similarity_test(original, translated):
    try:
        response = requests.post(SIMILARITY_API, headers = {
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
        
#Given a model, a question list and a language, it calculates the average similarity scores between the original questions from the question list and their translated version in the provided language
def test_system_translation(model, question_list, language):
    similarity = []
    for question in question_list:
        translation = ""
        back_transation = ""
        well_translated = True
        detector = LanguageDetectorBuilder.from_all_languages().build()
        try: 
            if model.model_name == DEEPL: #With DeepL
                translation = deepl_translation(model, question, language[LANGUAGES_CODE]) #From EN to the specified language 
                back_transation = deepl_translation(model, translation, "EN-US", language[LANGUAGES_CODE]) #From the specified language to EN
            else: #With LLM
                translation_attempt = 0 
                det_lan = ""
                while (translation == "" or translation == None or not well_translated) and translation_attempt < 3:
                    translation = model_translation(model, question, language[LANGUAGES]) 
                    #det = detect(translation) #detect(translation) 
                    det_lan = detector.detect_language_of(translation).name.lower() #detect(translation) 
                    corr_lan = language[LANGUAGES].lower() 
                    if det_lan != corr_lan and (corr_lan=="bosnian" and det_lan!="serbian" and det_lan!="croatian") and (corr_lan=="serbian" and det_lan!="bosnian" and det_lan!="croatian") and (corr_lan=="croatian" and det_lan!="serbian" and det_lan!="bosnian"): 
                        well_translated = False
                        logger.error(f"[({model.model_name}) Expected translation in {language[LANGUAGES]}, found {det_lan} instead]: {question} --> {translation}")
                    translation_attempt = translation_attempt + 1
                
                if well_translated:
                    translation_attempt = 0   
                    det_lan = ""
                    while (back_transation == "" or back_transation == None or not well_translated) and translation_attempt < 3:
                        back_transation = model_translation(model, translation)
                        #det = detect(back_transation)
                        det_lan = detector.detect_language_of(back_transation).name.lower()
                        
                        if det_lan == "english":
                            well_translated = True
                        else: 
                            logger.error(f"[({model.model_name}) Expected translation in English, found {det_lan} instead] {translation} --> {back_transation}")
                        translation_attempt = translation_attempt + 1
                
        except Exception as X:
            logger.error(f"test_system_translation: {X}")
            return None
        
        if translation == "" or "" == back_transation or not well_translated:
            similarity_score = 0
        else:
            similarity_score = similarity_test(question, back_transation) 
        similarity.append(similarity_score)
    return sum(similarity)/ len(similarity)

#Create the back_translation file, which contatins the average scores of back translation similarity test of every model in every language
def test_systems_translation_abilities(model_list):

    languages_list = []
    seen_languages = set()

    for country_name in COUNTRIES_FILE:
        for lang, code in zip(COUNTRIES_FILE[country_name][LANGUAGES], COUNTRIES_FILE[country_name][LANGUAGES_CODE]):
            if lang not in seen_languages:
                languages_list.append({LANGUAGES: lang, LANGUAGES_CODE: code})
                seen_languages.add(lang)

    columns, questions_list = load_create_question_list(languages_list)

    #Get the file
    csv_path = f"data/back_translation.csv"
    df = pd.read_csv(csv_path, sep=";") if os.path.exists(csv_path) else pd.DataFrame(columns=columns)

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
            df.loc[row_idx, language[LANGUAGES]] = round(score, 2) if score is not None else 0.0

            # Compute the average score, without considering Montenegrin
            lang_scores = pd.to_numeric(df.loc[row_idx, [lang[LANGUAGES] for lang in languages_list if lang[LANGUAGES] != 'Montenegrin']], errors="coerce")
            df.loc[row_idx, "avg_score"] = round(lang_scores.mean(), 2)
            
            df.to_csv(csv_path, sep=";", index=False)

# #Check models ability to support the langauges bit back translation 
model_list = [LLAMA31_70]
test_systems_translation_abilities(model_list)
