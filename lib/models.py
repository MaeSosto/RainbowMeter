from constants import *
import requests
from openai import OpenAI
#import google.generativeai as genai
from google import genai
import re
import deepl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

URL_OLLAMA_LOCAL = "http://localhost:11434/api"
URL_LMSTUDIO_LOCAL = "http://localhost:1234"
URL_DEEPSEEK = "https://api.deepseek.com"
URL_DEEPSEEK_POST = "https://api.deepseek.com/v1/chat/completions"

QWEN3_4 = "qwen/qwen3-4b-2507" #LMS
QWEN3_30 = "qwen/qwen3-30b-a3b-2507" #LMS
QWEN35_08 = "Qwen/Qwen3.5-0.8B" #HF
QWEN35_2_HF = "Qwen/Qwen3.5-2B" #HF
QWEN35_2_LMS = "qwen/qwen3.5-2b" #LMS
QWEN35_2 = "qwen3.5:2b" #HF
QWEN35_9_HF = "Qwen/Qwen3.5-9B" #HF
QWEN35_9_LMS = "qwen/qwen3.5-9b" #LMS
QWEN35_9 = "qwen3.5:9b" #Ollama
QWEN35_27_LMS = "qwen/qwen3.5-27b" #LMS
QWEN35_27 = "qwen3.5:27b" #Ollama
LlaMa32_3 ="llama3.2:3b" #Ollama
LlaMa31_8 = "llama3.1:8b" #Ollama
GEMMA3_4 = 'google/gemma-3-4b' #LMS
GEMMA3_12 = 'google/gemma-3-12b' #LMS
GEMMA3_27 = 'google/gemma-3-27b' 
MINISTRAL3_3 = 'mistralai/ministral-3-3b'
MINISTRAL3_8 = 'mistralai/ministral-3-8b'
MINISTRAL3_14 = 'mistralai/ministral-3-14b'
DEEPSEEKV32 = "deepseek-chat"
SONNET46 = "claude-sonnet-4-6"
GPT54 = 'gpt-5.4'
GPT54_MINI = 'gpt-5.4-mini'
GEMINI3_FLASH = 'gemini-3-flash-preview'
DEEPL = "DeepL" 

MODELS_LABELS = {
    QWEN3_4: "Qwen3 4B",
    QWEN3_30: "Qwen3 30B",
    QWEN35_08: "Qwen3.5 0.8B",
    QWEN35_2_LMS: "Qwen3.5 2B",
    QWEN35_2_HF: "Qwen3.5 2B",
    QWEN35_2: "Qwen3.5 2B",
    QWEN35_9_HF: "Qwen3.5 9B",
    QWEN35_9_LMS: "Qwen3.5 9B",
    QWEN35_9: "Qwen3.5 9B",
    QWEN35_27_LMS: "Qwen3.5 27B",
    QWEN35_27: "Qwen3.5 27B",
    LlaMa32_3: "LlaMa 3.2 3B",
    LlaMa31_8: "LlaMa 3.1 8B",
    GEMMA3_4: "Gemma 3 4B", 
    GEMMA3_12 : "Gemma 3 12B",
    GEMMA3_27 : "Gemma 3 27B",
    MINISTRAL3_3: "Ministral 3 3B",
    MINISTRAL3_8: "Ministral 3 8B",
    MINISTRAL3_14: "Ministral 3 14B",
    DEEPSEEKV32: "DeepSeek-V3.2",
    SONNET46: "Sonnet 4.6",
    DEEPL: "DeepL",
    GPT54 : 'GPT 4o',
    GPT54_MINI : 'GPT 5',
    GEMINI3_FLASH: 'Gemini 3 Flash'
}

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        
        self.func_initialize_model = {
            QWEN3_4: self._initialize_LMSStudio,
            QWEN3_30: self._initialize_LMSStudio,
            QWEN35_08: self._initialize_HuggingFace,
            QWEN35_2_LMS: self._initialize_LMSStudio,
            QWEN35_2_HF: self._initialize_HuggingFace,
            QWEN35_2: self._initialize_Ollama,
            QWEN35_9_HF: self._initialize_HuggingFace,
            QWEN35_9_LMS: self._initialize_LMSStudio,
            QWEN35_9: self._initialize_Ollama,
            QWEN35_27_LMS: self._initialize_LMSStudio,
            QWEN35_27: self._initialize_Ollama,
            LlaMa32_3: self._initialize_Ollama,
            LlaMa31_8: self._initialize_Ollama,
            GEMMA3_4: self._initialize_LMSStudio,
            GEMMA3_12: self._initialize_LMSStudio,
            GEMMA3_27: self._initialize_LMSStudio,
            MINISTRAL3_3: self._initialize_LMSStudio,
            MINISTRAL3_8: self._initialize_LMSStudio,
            MINISTRAL3_14: self._initialize_LMSStudio,
            DEEPSEEKV32: self.initialize_DeepSeek,
            SONNET46: self.initialize_Antrophic,
            GPT54: self.initialize_OpenAI, 
            GPT54_MINI: self.initialize_OpenAI,
            GEMINI3_FLASH: self._initialize_GoogleGenAI,
            DEEPL: self.initialize_Deepl
        }
        
        self.send_request = {
            QWEN3_4: self._request_LMSStudio,
            QWEN3_30: self._request_LMSStudio,
            QWEN35_08: self._request_HuggingFace,
            QWEN35_2_HF: self._request_HuggingFace,
            QWEN35_2_LMS: self._request_LMSStudio,
            QWEN35_2: self._request_Ollama,
            QWEN35_9_HF: self._request_HuggingFace,
            QWEN35_9_LMS: self._request_LMSStudio,
            QWEN35_9: self._request_Ollama,
            QWEN35_27_LMS: self._request_LMSStudio,
            QWEN35_27: self._request_Ollama,
            LlaMa32_3: self._request_Ollama,
            LlaMa31_8: self._request_Ollama,
            GEMMA3_4: self._request_LMSStudio,
            GEMMA3_12: self._request_LMSStudio,
            GEMMA3_27: self._request_LMSStudio,
            MINISTRAL3_3: self._request_LMSStudio,
            MINISTRAL3_8: self._request_LMSStudio,
            MINISTRAL3_14: self._request_LMSStudio,
            DEEPSEEKV32: self.request_OpenAi,
            SONNET46: self.request_Antrophic,
            GPT54: self.request_OpenAi, 
            GPT54_MINI: self.request_OpenAi,
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

    def _initialize_Ollama(self):
        try:
            response = requests.get(f"{URL_OLLAMA_LOCAL}/tags")
            if not(response.status_code == 200):
                logger.error(f"⚠️ Ollama server is not running")
                return True
            return False
        except requests.RequestException:
            logger.error(f"⚠️ Ollama server is not running")
            return True
    
    def _initialize_LMSStudio(self):
        try:
            response = requests.get(f"{URL_LMSTUDIO_LOCAL}/v1/models")
            if response.status_code != 200:
                logger.error("⚠️ LM Studio server is not running")
                return True
            return False
        except requests.RequestException:
            logger.error("⚠️ LM Studio server is not running")
            return True
    
    def _initialize_HuggingFace(self):
        logger.setLevel(logging.ERROR)
        try:
            if self.model_name == QWEN35_08 or self.model_name == QWEN35_2_HF:
                self.auto_processor = AutoProcessor.from_pretrained(self.model_name)
                self.auto_model = AutoModelForImageTextToText.from_pretrained(self.model_name)
                #self.auto_model = self.auto_model.to(device)
            else:
                self.auto_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.auto_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            return False
        except Exception as X:
            logger.error(f"⚠️ Hugging Face model {self.model_name} cannot be initialized")
        return True
    
    def _request_Ollama(self):
        try:
            r = requests.post(
                f"{URL_OLLAMA_LOCAL}/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "prompt": self.prompt,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "stream": False,
                    "max_new_tokens": 500
                }
            )

            if r.status_code != 200:
                # data = r.json()
                # response = data["content"]["error"]
                logger.error(f"⚠️ Ollama ERROR: {response.reason}")
                return None

            
            data = r.json()
            response = data["response"]

            if response is None or response == "":
                logger.error("_request_ollama: empty response")
                return None

            return response

        except Exception as X:
            logger.error(f"_request_ollama: {X}")
            return None
    import requests

    def _request_LMSStudio(self):
        try:
            response = requests.post(
                f"{URL_LMSTUDIO_LOCAL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "stream": False,
                    "max_new_tokens": 500
                },
                timeout=60  # in seconds
            )

            if response is None or response == "":
                logger.error(f"_request_lmstudio: empty response")
                return None

            data = response.json()
            response = data["choices"][0]["message"]["content"]
            return response

        except Exception as e:
            logger.error(f"_request_lmstudio error: {str(e)}")
            return None
    
    def request_GoogleGenAI(self):
        logger.setLevel(logging.ERROR)
        try:
            response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self.prompt,
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
            return completion.choices[0].message.content
        except Exception as X:
            logger.error(f"_request_open_ai: {X}")
            return None
    
    def request_Antrophic(self):
        logger.setLevel(logging.ERROR)
        try:
            message = self.client.messages.create(
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                    }
                ],
                model=self.model_name,
            )
            resp =  message.content[0].text
            
            logger.setLevel(logging.INFO)
            return resp
        except Exception as X:
            logger.error(f"request_Antrophic: {X}")
            return None
        
    def _request_HuggingFace(self):
        logger.setLevel(logging.ERROR)
        try:
            # if self.model_name == EUROLLM_9:
            #     #EuroLLM
            #     inputs = self.auto_tokenizer(self.prompt, return_tensors="pt")
            #     gen_tokens = self.auto_model.generate(**inputs, max_new_tokens=500)
            #     out = self.auto_tokenizer.decode(gen_tokens[0])
            
            # else:
            #print("question")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt}
                    ]
                },
            ]
            inputs = self.auto_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                #pad_token_id=self.auto_processor.eos_token_id,
            ).to(self.auto_model.device)
            
            outputs = self.auto_model.generate(**inputs, max_new_tokens=40)
            out_ = self.auto_processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            out_ = extract_model_answer(out_)
            #print("answer")
            return out_
                # inputs = self.auto_tokenizer(self.prompt, return_tensors="pt")
                # outputs = self.auto_model.generate(**inputs, max_new_tokens=500)

                # out = self.auto_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # out_ = extract_model_answer(out)
            
            # else self.model_name == COMMAND_R1:
            #     #Command R1
            #     inputs = self.auto_tokenizer.apply_chat_template(
            #         [{"role": "user", "content": self.prompt}],
            #         tokenize=True,
            #         add_generation_prompt=True,
            #         return_tensors="pt"
            #     )

            #     gen_tokens = self.auto_model.generate(
            #         **inputs,   
            #         max_new_tokens=500,
            #         do_sample=True
            #     )

            #     out = self.auto_tokenizer.decode(gen_tokens[0])
            #     out_ = extract_model_answer(out)
            #     return out_
            
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
