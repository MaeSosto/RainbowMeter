from constants import *
import requests
from openai import OpenAI
#import google.generativeai as genai
#from google import genai
#import google.generativeai as genai
from google import genai
from google.genai import types
import re
import deepl
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, pipeline
from dotenv import load_dotenv
from anthropic import Anthropic
from transformers import GenerationConfig

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

load_dotenv()
MAX_GENERATION_TOKEN = 100

URL_OLLAMA_LOCAL = "http://localhost:11434/api"
URL_LMSTUDIO_LOCAL = "http://localhost:1234"
URL_DEEPSEEK = "https://api.deepseek.com"
URL_DEEPSEEK_POST = "https://api.deepseek.com/v1/chat/completions"

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        
        self.func_initialize_model = {
            QWEN35_2: self.initialize_HuggingFace,
            QWEN35_9: self.initialize_HuggingFace,
            QWEN35_27: self.initialize_HuggingFace,
            LLAMA32_3: self.initialize_HuggingFace,
            LLAMA31_8: self.initialize_HuggingFace,
            LLAMA31_70: self.initialize_HuggingFace,
            DEEPSEEKV32: self.initialize_DeepSeek,
            SONNET46: self.initialize_Antrophic,
            GPT54: self.initialize_OpenAI, 
            GEMINI3_FLASH: self._initialize_GoogleGenAI,
            DEEPL: self.initialize_Deepl
        }
        
        self.send_request = {
            QWEN35_2: self.request_HuggingFace,
            QWEN35_9: self.request_HuggingFace,
            QWEN35_27: self.request_HuggingFace,
            LLAMA32_3: self.request_HuggingFace,
            LLAMA31_8: self.request_HuggingFace,
            LLAMA31_70: self.request_HuggingFace,
            DEEPSEEKV32: self.request_OpenAi,
            SONNET46: self.request_Antrophic,
            GPT54: self.request_OpenAi, 
            GEMINI3_FLASH: self.request_GoogleGenAI
        }
        
    def initialize_model(self):
        if self.model_name in self.func_initialize_model: 
            err = self.func_initialize_model[self.model_name]()
            return err
        return False

    def initialize_Deepl(self):
        try: 
            self.client = deepl.DeepLClient(os.getenv('DEEPL_API_KEY'))
            return False
        except Exception as X:
            logger.error(f"_initialize_deepl: {X}")
            return True

    def _initialize_GoogleGenAI(self):
        logger.setLevel(logging.ERROR) 
        api_key = os.getenv('GENAI_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ GENAI_API_KEY is missing")
            return True
        # genai.configure(api_key=api_key) 
        # self.client = genai.GenerativeModel(self.model_name)
        self.client = genai.Client(api_key=api_key)
        logger.setLevel(logging.INFO)
        return False

    def initialize_OpenAI(self): 
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ OPENAI_API_KEY is missing")
            return True
        self.client = OpenAI(api_key=api_key)
        return False
    
    def initialize_Antrophic(self):
        api_key = os.getenv('CLAUDE_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ CLAUDE_API_KEY is missing")
            return True
        self.client = Anthropic(api_key=api_key)
        return False

    def initialize_DeepSeek(self): 
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ DEEPSEEK_API_KEY is missing")
            return True
        self.client = OpenAI(api_key=api_key, base_url=URL_DEEPSEEK)
        return False
    
    def initialize_HuggingFace(self):
        logger.setLevel(logging.ERROR)
        try:
            if "qwen" in self.model_name.lower():
                self.auto_processor = AutoProcessor.from_pretrained(self.model_name)
                self.auto_model = AutoModelForImageTextToText.from_pretrained(self.model_name)
            
            else:
                self.auto_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.auto_tokenizer,
                    dtype=torch.bfloat16,
                    device_map="auto"
                )
            return False
        except Exception as X:
            logger.error(f"⚠️ Hugging Face model {self.model_name} cannot be initialized: {X}")
        return True
    
    def request_GoogleGenAI(self):
        logger.setLevel(logging.ERROR)
        try:
            response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self.prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="minimal")
                    )
                )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_GENERATION_TOKEN,
                    thinking_config=types.ThinkingConfig(
                        thinkingBudget=0
                    ),
                )
            )
            response = response.text 
            if response is None or response == "":
                logger.error(f"request_GoogleGenAI: empty response")
                return None
            logger.setLevel(logging.INFO)
            return response
        except Exception as X:
            logger.error(f"_request_GoogleGenAI: {X}")
            return None

    def request_OpenAi(self):
        logger.setLevel(logging.ERROR)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name, 
                store=True,
                messages=[{
                    "role": "user", 
                    "content": self.prompt
                }]
            )
            logger.setLevel(logging.INFO)
            response = completion.choices[0].message.content
            if response is None or response == "":
                    logger.error(f"request_OpenAi: empty response")
                    return None
            return response
        except Exception as X:
            logger.error(f"request_OpenAi: {X}")
            return None
    
    def request_Antrophic(self):
        logger.setLevel(logging.ERROR)
        try:
            message = self.client.messages.create(
                max_tokens=MAX_GENERATION_TOKEN,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                    }
                ],
                model=self.model_name,
            )
            response =  message.content[0].text
            
            logger.setLevel(logging.INFO)
            if response is None or response == "":
                logger.error(f"request_Antrophic: empty response")
                return None
            return response
        except Exception as X:
            logger.error(f"request_Antrophic: {X}")
            return None
        
    def request_HuggingFace(self):
        logger.setLevel(logging.ERROR)
        try:
            if "qwen" in self.model_name.lower():
                messages = [{"role": "user", "content": [{"type": "text", "text": self.prompt}]},]
                
                inputs = self.auto_processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                    #pad_token_id=self.auto_processor.eos_token_id,
                ).to(self.auto_model.device)
                
                outputs = self.auto_model.generate(
                    **inputs,
                    max_new_tokens=MAX_GENERATION_TOKEN,
                    max_length=None,
                    eos_token_id=self.auto_processor.tokenizer.eos_token_id
                )
                answer = self.auto_processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
                answer = extract_model_answer(answer)
            else:
                generation_config = GenerationConfig(
                    do_sample=True,
                    max_new_tokens=MAX_GENERATION_TOKEN,
                    pad_token_id=self.auto_tokenizer.eos_token_id,
                )

                answer = self.pipeline(
                    self.prompt,
                    generation_config=generation_config,
                )
                answer = answer[0]["generated_text"]
                answer = extract_model_answer(answer)
                
            return answer            
        except Exception as X:
            logger.error(f"_request_huggingface: {X}")
            return None
    
    def call_model(self, prompt):
        self.prompt = prompt
        try: 
            res = self.send_request[self.model_name]()
            return res
        except Exception as X:
            logger.error(X)
            return None

def extract_model_answer(text):
    # Split on the chatbot marker
    parts = text.split("<|CHATBOT_TOKEN|>")
    if len(parts) < 2:
        answer = text
    else:
        answer = parts[-1]

    # Remove specific unwanted ending patterns
    answer = answer.replace("<|im_end|>\n<|endoftext|>", "")
    answer = answer.replace("<|im_end|><|endoftext|>", "")

    # Remove any remaining special tokens like <|...|>
    answer = re.sub(r"<\|.*?\|>", "", answer)

    return answer.strip()


#Translate the given text in the specified language using DeepL    
def deepl_translation(model, question, target_lang = "EN-US", source_lang = "EN"):
    if target_lang == "pt":
        target_lang = "pt-pt"
    if target_lang == "en":
        target_lang = "EN-GB"
    if question == "":
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
        return translation
    except Exception as X:
        logger.error(f"translate: {X}")
        return None
        #return None
