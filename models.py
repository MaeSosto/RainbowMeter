from lib import *
import requests
import os

RESPONSE_CHOICE = ["STRONGLY DISAGREE", "DISAGREE", "AGREE", "STRONGLY AGREE"]

def callModel(prompt, modelName):
    response = ollamaRequest(
        prompt=prompt,
        modelName=modelName 
    )
    response = _clearResponse(response)
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

def _getPrompt(prompt_template, text):
    label = prompt_template["labels"][0]
    format_instructions = f'{{ "answer": "{label}" }}'
    template = prompt_template['prompt_template']

    # Assuming 'text' is passed into the function or available
    prompt_template = template.format(
        text=text,
        format_instructions=format_instructions
    )
    return prompt_template

def _clearResponse(response):
    try:
        # Step 1: Remove markdown formatting (e.g. ```json ... ```)
        response = re.sub(r"^```(?:json)?\n|```$", "", response.strip())
        response = json.loads(response)
        return response["answer"]
        # for choice in RESPONSE_CHOICE:
        #     answ = response["answer"] 
        #     if answ == choice:
        #         return choice
        # breakpoint
        # return ""
    except Exception as X:
        breakpoint