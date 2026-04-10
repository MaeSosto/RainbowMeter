from constants import *
import requests
from openai import OpenAI
import google.generativeai as genai
import re
import deepl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

URL_OLLAMA_LOCAL = "http://localhost:11434/api"
URL_LMSTUDIO_LOCAL = "http://localhost:1234"
URL_DEEPSEEK = "https://api.deepseek.com"

QWEN3_4 = "qwen/qwen3-4b-2507" #LMS
QWEN3_30 = "qwen/qwen3-30b-a3b-2507" #LMS
QWEN35_08 = "Qwen/Qwen3.5-0.8B" #HF
QWEN35_2 = "Qwen/Qwen3.5-2B" #HF
QWEN35_9 = "qwen3.5:9b" #Ollama
QWEN35_27 = "qwen3.5:27b" #Ollama
LlaMa31_8 = "llama3.1:8b" #Ollama
GEMMA3_4 = 'google/gemma-3-4b' #LMS
GEMMA3_12 = 'google/gemma-3-12b' #LMS
GEMMA3_27 = 'google/gemma-3-27b' 
MINISTRAL3_3 = 'mistralai/ministral-3-3b'
MINISTRAL3_8 = 'mistralai/ministral-3-8b'
MINISTRAL3_14 = 'mistralai/ministral-3-14b'
DEEPSEEKR1_1_5 = 'deepseek-r1:1.5b'
DEEPSEEKR1_8 = 'deepseek-r1:8b'
DEEPSEEKR1_32 = 'deepseek-r1:32b'
DEEPSEEKR1_32_DISTILL = 'deepseek/deepseek-r1-distill-qwen-32b'
GPT4 = 'gpt-4o'
GPT5 = 'gpt-5'
DEEPL = "DeepL"
EUROLLM_9 = "utter-project/EuroLLM-9B"
COMMAND_R1 = "CohereLabs/c4ai-command-r-v01"

MODELS_LABELS = {
    QWEN3_4: "Qwen3 4B",
    QWEN3_30: "Qwen3 30B",
    QWEN35_08: "Qwen3.5 0.8B",
    QWEN35_2: "Qwen3.5 2B",
    QWEN35_9: "Qwen3.5 9B",
    QWEN35_27: "Qwen3.5 27B",
    LlaMa31_8: "LlaMa 3.1 8B",
    GEMMA3_4: "Gemma 3 4B", 
    GEMMA3_12 : "Gemma 3 12B",
    GEMMA3_27 : "Gemma 3 27B",
    MINISTRAL3_3: "Ministral 3 3B",
    MINISTRAL3_8: "Ministral 3 8B",
    MINISTRAL3_14: "Ministral 3 14B",
    DEEPSEEKR1_1_5: "DeepSeek R1 1.5B",
    DEEPSEEKR1_8: "DeepSeek R1 8B",
    DEEPSEEKR1_32: "DeepSeek R1 32B",
    DEEPSEEKR1_32_DISTILL: "DeepSeek R1 DIST 32B",
    DEEPL: "DeepL",
    EUROLLM_9: "EuroLLM 9B",
    COMMAND_R1: "Command R1",
    GPT4 : 'GPT 4o',
    GPT5 : 'GPT 5'
}

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        
        self.func_initialize_model = {
            QWEN3_4: self._initialize_LMSStudio,
            QWEN3_30: self._initialize_LMSStudio,
            QWEN35_08: self._initialize_HuggingFace,
            QWEN35_2: self._initialize_HuggingFace,
            QWEN35_9: self._initialize_Ollama,
            QWEN35_27: self._initialize_Ollama,
            LlaMa31_8: self._initialize_Ollama,
            GEMMA3_4: self._initialize_LMSStudio,
            GEMMA3_12: self._initialize_LMSStudio,
            GEMMA3_27: self._initialize_LMSStudio,
            MINISTRAL3_3: self._initialize_LMSStudio,
            MINISTRAL3_8: self._initialize_LMSStudio,
            MINISTRAL3_14: self._initialize_LMSStudio,
            DEEPSEEKR1_1_5: self._initialize_Ollama,
            DEEPSEEKR1_8: self._initialize_Ollama,
            DEEPSEEKR1_32: self._initialize_Ollama,
            DEEPSEEKR1_32_DISTILL: self._initialize_LMSStudio,
            GPT4: self._initialize_OpenAI, 
            GPT5: self._initialize_OpenAI, 
            DEEPL: self._initialize_deepl,
            EUROLLM_9: self._initialize_HuggingFace,
            COMMAND_R1: self._initialize_HuggingFace
        }
        
        self.send_request = {
            QWEN3_4: self._request_LMSStudio,
            QWEN3_30: self._request_LMSStudio,
            QWEN35_08: self._request_HuggingFace,
            QWEN35_2: self._request_HuggingFace,
            QWEN35_9: self._request_Ollama,
            QWEN35_27: self._request_Ollama,
            LlaMa31_8: self._request_Ollama,
            GEMMA3_4: self._request_LMSStudio,
            GEMMA3_12: self._request_LMSStudio,
            GEMMA3_27: self._request_LMSStudio,
            MINISTRAL3_3: self._request_LMSStudio,
            MINISTRAL3_8: self._request_LMSStudio,
            MINISTRAL3_14: self._request_LMSStudio,
            DEEPSEEKR1_1_5: self._request_Ollama,
            DEEPSEEKR1_8: self._request_Ollama,
            DEEPSEEKR1_32: self._request_Ollama,
            DEEPSEEKR1_32_DISTILL: self._request_LMSStudio,
            GPT4: self._request_OpenAi, 
            GPT5: self._request_OpenAi, 
            EUROLLM_9: self._request_HuggingFace,
            COMMAND_R1: self._request_HuggingFace,
        }
        
    def initialize_model(self):
        if self.model_name in self.func_initialize_model: 
            err = self.func_initialize_model[self.model_name]()
            return err
        return False

    def _initialize_deepl(self):
        try: 
            self.client = deepl.DeepLClient(os.getenv('DEEPL_API_KEY'))
            return False
        except Exception as X:
            logger.error(f"_initialize_deepl: {X}")
            return True

    def _initialize_Gemini(self): 
        api_key = os.getenv('GENAI_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ GENAI_API_KEY is missing")
            return True
        genai.configure(api_key=api_key) 
        self.client = genai.GenerativeModel(self.model_name)
        return False

    def _initialize_OpenAI(self): 
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ OPENAI_API_KEY is missing")
            return True
        self.client = OpenAI(api_key=api_key)
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
    
    def _initialize_DeepSeeek(self): 
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ DEEPSEEK_API_KEY is missing")
            return True
        self.client = OpenAI(api_key=api_key, base_url=URL_DEEPSEEK)
        return False
    
    def _initialize_HuggingFace(self):
        logger.setLevel(logging.ERROR)
        try:
            if self.model_name == QWEN35_08 or self.model_name == QWEN35_2:
                self.auto_processor = AutoProcessor.from_pretrained(self.model_name)
                self.auto_model = AutoModelForImageTextToText.from_pretrained(self.model_name)
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
                logger.error(f"⚠️ Ollama ERROR: {response}")
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
    
    def _request_gemini(self):
        try:
            return self.client.generate_content(self.prompt).text
        except Exception as X:
            logger.error(f"_request_gemini: {X}")
            return None

    def _request_OpenAi(self):
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

    def _request_HuggingFace(self):
        logger.setLevel(logging.ERROR)
        try:
            if self.model_name == EUROLLM_9:
                #EuroLLM
                inputs = self.auto_tokenizer(self.prompt, return_tensors="pt")
                gen_tokens = self.auto_model.generate(**inputs, max_new_tokens=500)
                out = self.auto_tokenizer.decode(gen_tokens[0])
            
            elif self.model_name == QWEN35_08 or self.model_name == QWEN35_2:
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
