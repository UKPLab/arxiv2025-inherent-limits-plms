import os
import torch
import openai
import config
import time

openai.api_key = config.openai_api_key
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_api_response(prompt):
    client = openai.OpenAI(api_key=openai.api_key,
                           )
    try:
        response = client.chat.completions.create(
            model=config.api_model,
            messages=[{"role": "user", "content": prompt}]
            )
        response = response.choices[0].message.content
    except openai.RateLimitError:
        print("Rate limit exceeded. Waiting 10 seconds...")
        time.sleep(10)
    return response