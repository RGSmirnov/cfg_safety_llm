# call_gpt

import openai
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
if OPENAI_API_KEY == '':
    OPENAI_API_KEY = "sk-..."
CLIENT = openai.OpenAI(api_key = OPENAI_API_KEY)

def call_openai_api(message_log, model = 'gpt-4o-mini', max_new_tokens = 1024, temperature = 0.0):
    response = CLIENT.chat.completions.create(
            model=model, 
            messages=message_log,   
            max_tokens=max_new_tokens,        
            temperature=temperature,      
        )
    for choice in response.choices:
        if "text" in choice:
            return choice.text
    response = response.choices[0].message.content
    return response