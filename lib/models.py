from lib.prompt import *
from lib.constants import *
import requests
from openai import OpenAI

def initialize_gpt(model_name=None): 
    key = os.getenv('OPENAI_API_KEY')
    return OpenAI(api_key=key)

def _ollama_request (prompt, modelName, client = None):
    URL_OLLAMA_LOCAL = "http://localhost:11434/api/generate"
    try:
        response = requests.post(URL_OLLAMA_LOCAL, headers={
                "Content-Type": 'application/json'
            }, 
            json={
                "model": modelName,
                "prompt": prompt,
                "messages": [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ],
                "options":{
                    "temperature":0
                },
                    "stream": False
        })
        tmp = response.json()
        tmp = tmp['response']
        return tmp
    except Exception as X:
        logger.error(f"_ollama_request: {X}")
        return None
    
def _gpt_request(prompt, modelName, client):
    console.setLevel(logging.ERROR)
    completion = client.chat.completions.create(
        model=modelName, store=True,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    console.setLevel(logging.INFO)
    return completion.choices[0].message.content

initialize_models = {
    GPT4: initialize_gpt, 
    GPT4_MINI: initialize_gpt,
}

request_models = {
    LLAMA3: _ollama_request, 
    LLAMA3_70B: _ollama_request, 
    GEMMA3: _ollama_request,
    GEMMA3_27B: _ollama_request,
    GPT4: _gpt_request, 
    GPT4_MINI: _gpt_request,
}

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = initialize_models[model_name]() if model_name in initialize_models else None
        
    def call_model(self, prompt):
        
        response = request_models[self.model_name](
            prompt=prompt,
            modelName=self.model_name,
            client = self.client
        )
        response = clear_response(response)
        return response
