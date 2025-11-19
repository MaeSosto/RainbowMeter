from lib.constants import *
import requests
from openai import OpenAI
import google.generativeai as genai
import re

URL_OLLAMA_LOCAL = "http://localhost:11434/api"
URL_DEEPSEEK = "https://api.deepseek.com"

class Model:
    def __init__(self, name):
        self.name = name
        
        self.func_initialize_model = {
            LLAMA3: self._initialize_Ollama, 
            LLAMA3_70B: self._initialize_Ollama,
            LLAMA4: self._initialize_Ollama,
            GEMMA3: self._initialize_Ollama,
            GEMMA3_27B: self._initialize_Ollama, 
            GPT4: self._initialize_GPT, 
            GPT4_MINI: self._initialize_GPT,
            DEEPSEEK: self._initialize_Ollama,
            DEEPSEEK_671B: self._initialize_DeepSeeek,
            GEMINI_2_0_FLASH: self._initialize_Gemini, 
            GEMINI_2_0_FLASH_LITE: self._initialize_Gemini,
        }
        
        self.send_request = {
            LLAMA3: self._request_ollama, 
            LLAMA3_70B: self._request_ollama,
            LLAMA4: self._request_ollama,  
            GEMMA3: self._request_ollama,
            GEMMA3_27B: self._request_ollama, 
            DEEPSEEK: self._request_ollama,
            DEEPSEEK_671B: self._request_open_ai,
            GPT4: self._request_open_ai, 
            GPT4_MINI: self._request_open_ai,
            GEMINI_2_0_FLASH: self._request_gemini, 
            GEMINI_2_0_FLASH_LITE: self._request_gemini,
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
                "options": {"temperature": 0},
                "stream": False
            })
            response = response.json()['response']
            if response == None or response == "":
                logger.error(f"_request_ollama: {response}")
                return None
            return response
        
        except Exception as X:
            logger.error(f"_request_ollama: {response['text']}")
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
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0
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
        