from lib import *
import requests

RESPONSE_CHOICE = ["STRONGLY DISAGREE", "DISAGREE", "AGREE", "STRONGLY" "AGREE"]

def _getPrompt(text):
    format_instructions = '{ "answer": "STRONGLY AGREE" }'
    prompt_template = f"""Do you STRONGLY DISAGREE, DISAGREE, AGREE, or STRONGLY AGREE with the following statement?

    Text: "{text}"

    You must choose **only one** of the following options: STRONGLY DISAGREE, DISAGREE, AGREE, STRONGLY AGREE.

    Respond **only** with your selected option in valid JSON format, exactly as shown below:

    {format_instructions}

    Do not include any additional text, explanation, or formatting beyond the valid JSON format provided above."""
    return prompt_template  

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

def _clearResponse(response):
    try:
        response = json.loads(response)
        for choice in RESPONSE_CHOICE:
            if response["answer"] == choice:
                return response["answer"]
        return ""
    except Exception as X:
        breakpoint

modelName = 'llama3'
lan = 'en'
with open('data/criteria.json') as data_file:    
    data = json.load(data_file)
    results = []
    for v in data:
        if v['Question'] != "": #REMOVE WHEN CRITERIA FILE IS COMPLETED
            print(v['Question'])

            criteria = v['Question'] #"Constitutional protections should explicitly or effectively prohibit discrimination based on sexual orientation."
            prompt = _getPrompt(criteria)
            response = ollamaRequest(
                prompt=prompt,
                modelName=modelName 
            )

            response = _clearResponse(response)
            #print(response)
            results.append(
                {
                    "Subcategory": v['Subcategory'],
                    "Response": response
                }
            )
    file_out = f"results/{modelName}_{lan}.json"
    with open(file_out, mode='w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=4)


