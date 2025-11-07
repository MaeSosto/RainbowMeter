from lib.constants import *

prompt_types = ["Support", "Opposition"] #, "Fact Checking"] #Let's stay with just this now 
folder_result = "results_for_analysis/"
folder_language_scenario = "language_scenario/"
path_rainbow_meter = "data/rainbow_meter/"

class Prompt:
    def __init__(self, country):
        self.country = country
        self.prompt_template, self.retry_prompt_template, self.country.labels = self._get_prompt_templates()

    def _get_prompt_templates(self):
        prompt_template = f'data/prompts_data/prompts_template_{self.country.language}.json'
        prompt_template = open(prompt_template)
        prompt_template = json.load(prompt_template)
        return prompt_template["prompt_template"], prompt_template["retry_prompt_template"], prompt_template["labels"]
    
    def get_standard_prompt(self, text):
        format_instructions = f'{{ "answer": "{self.country.labels[0]}" }}'

        prompt_template = self.prompt_template.format(
            text=text,
            format_instructions=format_instructions
        )
        return prompt_template