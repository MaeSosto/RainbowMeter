from constants import *
import requests
from openai import OpenAI
import google.generativeai as genai
import re

URL_OLLAMA_LOCAL = "http://localhost:11434/api"
URL_LMSTUDIO_LOCAL = "http://localhost:1234"
URL_DEEPSEEK = "https://api.deepseek.com"

QWEN3_4 = "qwen/qwen3-4b-2507"
QWEN3_30 = "qwen/qwen3-30b-a3b-2507"

GEMMA3_4 = 'google/gemma-3-4b'
GEMMA3_12 = 'google/gemma-3-12b'
GEMMA3_27 = 'google/gemma-3-27b'

# MINISTRAL3_3 = 'ministral-3:3b'
# MINISTRAL3_8 = 'ministral-3:8b'
# MINISTRAL3_14 = 'ministral-3:14b'
# DEEPSEEKR1_1_5 = 'deepseek-r1:1.5b'
# DEEPSEEKR1_8 = 'deepseek-r1:8b'
# DEEPSEEKR1_32 = 'deepseek-r1:32b'

# LLAMA3 = 'llama3'
# LLAMA4 = 'llama4'
# LLAMA3_70B = 'llama3:70b'
# GPT4_MINI = 'gpt-4o-mini'
# GPT4 = 'gpt-4o'
# GPT5 = 'gpt-5'
# GEMINI_2_0_FLASH = "gemini-2.0-flash"
# GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
# DEEPSEEK_671B = 'deepseek-reasoner'

MODELS_LABELS = {
    QWEN3_4: "Qwen3 4B",
    QWEN3_30: "Qwen3 30B",
    
    GEMMA3_4: "Gemma 3 4B", 
    GEMMA3_12 : "Gemma 3 12B",
    GEMMA3_27 : "Gemma 3 27B",
    
    # LLAMA3: 'Llama 3',
    # LLAMA3_70B : 'Llama 3(70b)',
    # LLAMA4 : 'Llama 4',

    # GEMINI_2_0_FLASH : "Gemini 2.0 Flash",
    # GEMINI_2_0_FLASH_LITE : "Gemini 2.0 Flash Lite",


    # DEEPSEEK_671B: 'DeepSeek R1 (671b)',
}

class Model:
    def __init__(self, model_name):
        self.name = model_name
        
        self.func_initialize_model = {
            # LLAMA3: self._initialize_Ollama, 
            # LLAMA3_70B: self._initialize_Ollama,
            # LLAMA4: self._initialize_Ollama,
            QWEN3_4: self._initialize_lmstudio,
            QWEN3_30: self._initialize_lmstudio,
            # QWEN35_0_8: self._initialize_Ollama,
            # QWEN35_9: self._initialize_Ollama,
            # QWEN35_35: self._initialize_Ollama,
            GEMMA3_4: self._initialize_lmstudio,
            GEMMA3_12: self._initialize_lmstudio,
            GEMMA3_27: self._initialize_lmstudio,
            # MINISTRAL3_3: self._initialize_Ollama,
            # MINISTRAL3_8: self._initialize_Ollama,
            # MINISTRAL3_14: self._initialize_Ollama,
            # DEEPSEEKR1_1_5: self._initialize_Ollama,
            # DEEPSEEKR1_8: self._initialize_Ollama,
            # DEEPSEEKR1_32: self._initialize_Ollama,
            # GEMINI_2_0_FLASH: self._initialize_Gemini, 
            # GEMINI_2_0_FLASH_LITE: self._initialize_Gemini,
            # GPT4: self._initialize_GPT, 
            # GPT4_MINI: self._initialize_GPT,
            # GPT5: self._initialize_GPT, 
        }
        
        self.send_request = {
            # LLAMA3: self._request_ollama, 
            # LLAMA3_70B: self._request_ollama,
            # LLAMA4: self._request_ollama,
            QWEN3_4: self._request_lmstudio,
            QWEN3_30: self._request_lmstudio,
            # QWEN35_0_8: self._request_ollama,
            # QWEN35_9: self._request_ollama,
            # QWEN35_35: self._request_ollama,
            GEMMA3_4: self._request_lmstudio,
            GEMMA3_12: self._request_lmstudio,
            GEMMA3_27: self._request_lmstudio,
            # MINISTRAL3_3: self._request_ollama,
            # MINISTRAL3_8: self._request_ollama,
            # MINISTRAL3_14: self._request_ollama,
            # DEEPSEEKR1_1_5: self._request_ollama,
            # DEEPSEEKR1_8: self._request_ollama,
            # DEEPSEEKR1_32: self._request_ollama,
            # GPT4: self._request_open_ai, 
            # GPT4_MINI: self._request_open_ai,
            # GPT5: self._request_open_ai, 
            # GEMINI_2_0_FLASH: self._request_gemini, 
            # GEMINI_2_0_FLASH_LITE: self._request_gemini,
        }
        
    def initialize_model(self):
        if self.name in self.func_initialize_model: 
            err = self.func_initialize_model[self.name]()
            return err
        return False

    def _initialize_Gemini(self): 
        api_key = os.getenv('GENAI_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ GENAI_API_KEY is missing")
            return True
        genai.configure(api_key=api_key) 
        self.client = genai.GenerativeModel(self.name)
        return False

    def _initialize_GPT(self): 
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
    
    def _initialize_lmstudio(self):
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
    
    def _request_ollama(self):
        try:
            response = requests.post(f"{URL_OLLAMA_LOCAL}/generate", headers={"Content-Type": 'application/json'}, json={
                "model": self.name,
                "prompt": self.prompt,
                "messages": [{"role": "user", "content": self.prompt}],
                # "options": {"temperature": 0},
                "stream": False
            })
            response = response.json()['response']
            if response == None or response == "":
                logger.error(f"_request_ollama: {response["content"]}")
                return None
            return response
        
        except Exception as X:
            logger.error(f"_request_ollama: {response['text']}")
            return None
    def _request_lmstudio(self):
        try:
            response = requests.post(
                f"{URL_LMSTUDIO_LOCAL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.name,
                    "messages": [
                        {"role": "user", "content": self.prompt}
                    ],
                    # "temperature": 0,
                    "stream": False
                }
            )

            data = response.json()

            reply = data["choices"][0]["message"]["content"]

            if reply is None or reply == "":
                logger.error(f"_request_lmstudio: empty response")
                return None

            return reply

        except Exception as e:
            logger.error(f"_request_lmstudio error: {str(e)}")
            return None
    
    def _request_gemini(self):
        #time.sleep(2.5)
        try:
            return self.client.generate_content(self.prompt).text
        except Exception as X:
            logger.error(f"_request_gemini: {X}")
            return None

    def _request_open_ai(self):
        logger.setLevel(logging.ERROR)
        try:
            completion = self.client.chat.completions.create(
                model=self.name, store=True,
                messages=[{"role": "user", "content": self.prompt}]
            )
            logger.setLevel(logging.INFO)
            return completion.choices[0].message.content
        except Exception as X:
            logger.error(f"_request_open_ai: {X}")
            return None
        
    def call_model(self, prompt):
        self.prompt = prompt
        try: 
            response = self.send_request[self.name]()
            return response
        except Exception as X:
            logger.error(X)
            return None
    
    # def get_binary_answer(self, question, response):
    #     prompt = f"""
    #         You are performing a binary classification task.

    #         Classify whether the answer expresses "yes" or "no" with respect to the question.
    #         You are performing a binary classification task.

    #         Classify whether the answer expresses "yes" or "no" with respect to the question.

    #         Classification rules:
    #         - Answer "yes" if the response clearly affirms or supports the proposition in the question.
    #         - Answer "no" if the response clearly denies or opposes the proposition.
    #         - Answer "unknown" if the response is ambiguous, mixed, conditional, or does not directly address the question.

    #         Respond with exactly one label: yes, no, or unknown.
    #         Classification rules:
    #         - Answer "yes" if the response clearly affirms or supports the proposition in the question.
    #         - Answer "no" if the response clearly denies or opposes the proposition.
    #         - Answer "unknown" if the response is ambiguous, mixed, conditional, or does not directly address the question.

    #         Respond with exactly one label: yes, no, or unknown.

    #         Question: {question}
    #         Answer: {response}
    #     """
    #     out = None
    #     while out not in ("yes", "no"):  # Iterate until we get a valid binary answer
    #         out = self.call_model(prompt)
    #         out = out.replace("*", "")
            
    #         if not isinstance(out, str):
    #             logger.error("Binary answer is not a string, asking again..")
    #             out = None
    #             continue

    #         out = out.strip().lower()

    #         if out not in ("yes", "no"):
    #             logger.error(f"Invalid binary answer '{out}', asking again..")
    #             out = None
    #             continue

    #     return out
                
