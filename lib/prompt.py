from lib.constants import *
import re

PROMPT_TYPES = ["Support", "Opposition", "Fact"]
RESULT_FOLDER = "results/"
SCENARIO_LANGUAGE_FOLDER = "language_scenario/"
RAINBOW_METER_DATA_PATH = "data/rainbow_meter/"

class Prompt:
    def __init__(self, prompt_type, criteria, country):
        self.country = country
        self.prompt_type = prompt_type
        self.criteria_text = criteria[prompt_type]
        self.stance_type = "FC" if prompt_type == "Fact" else "BS"
        self.prompt_template = self._get_prompt_templates()
        self.format_instructions = f'{{ "answer": "{self.prompt_template[self.stance_type]['labels'][0]}" }}'

    def _get_prompt_templates(self):
        prompt_template = f'data/prompts_data/prompt_template_{self.country.language}.json'
        prompt_template = open(prompt_template)
        prompt_template = json.load(prompt_template)
        return prompt_template
        #self.BS_standard_template, self.retry_prompt_template, self.country.labels =  prompt_template["prompt_template"], prompt_template["retry_prompt_template"], prompt_template["labels"]
    
    def get_standard_prompt(self,):
        prompt_template = self.prompt_template[self.stance_type]["standard"].format(
            text=self.criteria_text,
            format_instructions=self.format_instructions
        )
        return prompt_template
    
    def get_retry_prompt(self, chat_history):
        prompt_template = self.prompt_template[self.stance_type]["retry"].format(
            chat_history = chat_history,
            text=self.criteria_text,
            format_instructions=self.format_instructions
        )
        return prompt_template
    
    def retry_prompt(self, chat_history):
        prompt = f"""Previous answer: {chat_history} \n
            Answer only with "yes" or "no": {self.criteria_text}"""
        return prompt
    
    def check_response(self, response):
        try:
            response = re.sub(r"^```(?:json)?\n|```$", "", response.strip())
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = json.loads(response)
            if response["answer"] in self.prompt_template[self.stance_type]["labels"]:
                return response["answer"]
            else:
                return None
        except Exception as X:
            return None