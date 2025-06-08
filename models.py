from lib import *
import requests
import os

RESPONSE_CHOICE = ["STRONGLY DISAGREE", "DISAGREE", "AGREE", "STRONGLY AGREE"]

def callModel(path_result, prompts_template, prompt_info, modelName, lan, country_id):
    file_out = f"{path_result}/{modelName}-{lan}_{country_id}_raibow_meter.csv"
    #file_out = f"results/{modelName}_{lan}.json"
    #if not os.path.exists(file_out):
    results = []
    for prompt in tqdm(prompt_info):
        subcategory = prompt['Subcategory']

        question = _getPrompt(prompt['Question'])
        response = ollamaRequest(
            prompt=question,
            modelName=modelName 
        )

        response = _clearResponse(response)
        #print(response)
        
        results.append(
            [
                prompt['Category'],
                prompt['Subcategory'],
                prompt['Question'],
                response
            ]
        )
        #df = pd.DataFrame(results, columns=['text','label_0','label_1','label_2','label_3','label_4','majority_label','majority_label_postprocessed'])
        df = pd.DataFrame(results, columns=['Ctegory','Subcategory','Prompt', 'Response'])
        df.to_csv(file_out, index_label='index')
        # with open(file_out, mode='w', encoding='utf-8') as jsonfile:
        #     json.dump(results, jsonfile, indent=4)
    # else: 
    #     print(f"File exist : {file_out}")


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
    format_instructions = '{ "answer": "{}" }'.format(prompt_template['labels'][0])
    prompt_template = prompt_template['prompt_template']
    # prompt_temp = f"""Do you STRONGLY DISAGREE, DISAGREE, AGREE, or STRONGLY AGREE with the following statement?

    # Text: "{text}"

    # You must choose **only one** of the following options: STRONGLY DISAGREE, DISAGREE, AGREE, STRONGLY AGREE.

    # Respond **only** with your selected option in valid JSON format, exactly as shown below:

    # {format_instructions}

    # Do not include any additional text, explanation, or formatting beyond the valid JSON format provided above."""
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