from models import *

class Translator:
    
    def __init__(self, language):
        self.language = language
        self.model = model

    def check_support_language(self, language):
            text = "This is a test text to check if the model supports the language."
            format_instructions = f'{{ "translation": "{text}" }}'
            translation_prompt_template = """Translate the following text to {language} and provide the translation in valid JSON format only as shown below:\n\nText: \"{text}\"\n\nProvide only the translation without any explanation in valid JSON format only as shown below:\n\n{format_instructions}"""


            # translation_prompt = PromptTemplate(
            #     template=translation_prompt_template,
            #     input_variables=["text", "language"],
            #     partial_variables={
            #         "format_instructions": parser.get_format_instructions()
            #     },
            # )

            # model_support_langaguage_chain = (
            #     translation_prompt | model | parser
            # )

            # try:
            #     model_output = model_support_langaguage_chain.invoke(
            #         {"text": text, "language": language}
            #     )
            #     translated_text = model_output["label"]
            #     time.sleep(1)
            #     model_output = model_support_langaguage_chain.invoke(
            #         {"text": translated_text, "language": "english"}
            #     )
            #     translated_back_text = model_output["label"]

            #     if (
            #         compute_cosine_similarity(
            #             similarity_model, text, translated_back_text
            #         )
            #         > 0.8
            #     ):
            #         return True
            # except Exception as e:
            #     print(e)
            #     return False

            # return False

obj = {
    "languages": [
        "Italian"
    ]
    }
language = obj["languages"][0]
model_name = LLAMA3

model = Model(model_name)
translator = Translator(language, model)
translator.check_support_language(language)
