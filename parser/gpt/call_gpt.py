from openai import OpenAI
import time
start = time.time()

import os
import openai
from openai import OpenAI

client = OpenAI(api_key=" ") # your api key

def request_with_try(prompt):
    # response = client.Completion.create(
    #   model="text-davinci-002",
    #   prompt=prompt,
    #   temperature=0,
    #   max_tokens=100,
    #   top_p=1,
    #   frequency_penalty=0.0,
    #   presence_penalty=0.0,
    #   stop=stop
    # )
    max_retries = 10
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
            model='gpt-3.5-turbo-16k',
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n = 1
            )
            break
        except:
            if i < max_retries - 1:  # 如果不是最后一次重试
                time.sleep(1)  # 等待一秒
                continue
            else:  # 如果是最后一次重试，那么重新引发错误
                raise

    return response.choices[0].message.content