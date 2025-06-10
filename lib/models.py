from prompt import *
import requests

RESPONSE_CHOICE = ["STRONGLY DISAGREE", "DISAGREE", "AGREE", "STRONGLY AGREE"]

def call_model(prompt, modelName):
    response = ollamaRequest(
        prompt=prompt,
        modelName=modelName 
    )
    response = clear_response(response)
    return response


def ollamaRequest (prompt, modelName):
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
        #logger.error("ollamaRequest: "+str(X))
        breakpoint

