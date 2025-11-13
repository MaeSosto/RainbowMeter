from vllm import LLM, SamplingParams
from huggingface_hub import login 
import os 

login(os.getenv('HF_TOKEN'))

def prompt_llama3(prompt):

    llm = LLM(model="meta-llama/Meta-Llama-3-8B")
    sampling_params = SamplingParams(
        temperature=0
    )
    response = llm.generate(prompt, sampling_params)
    generated_text = response[0].outputs[0].text
    return generated_text


# Example usage
if __name__ == "__main__":
    prompt = "Write a short story about a queer AI researcher."
    result = prompt_llama3(prompt)
    print(result)