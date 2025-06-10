import re
import json

def clear_response(response):
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

def get_standard_prompt(prompt_template, text):
    label = prompt_template["labels"][0]
    format_instructions = f'{{ "answer": "{label}" }}'
    template = prompt_template['prompt_template']

    # Assuming 'text' is passed into the function or available
    prompt_template = template.format(
        text=text,
        format_instructions=format_instructions
    )
    return prompt_template