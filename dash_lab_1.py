import os
import time
import json

from groq import Groq

client = Groq(api_key=os.environ.get("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"),)

with open('input.txt', 'r') as inp_file:
    inputs = inp_file.readlines()

outputs = []
src = "Grok"

for i in inputs:
    i = i.strip()
    time_sent = time.time()
    call = client.chat.completions.create(messages=[{"role": "user", "content": i}], model="llama3-8b-8192",)
    time_recvd = int(time.time())
    msg = call.choices[0].message.content
    
    obj = {
        "Prompt": i,
        "Message": msg,
        "TimeSent": time_sent,
        "TimeRecvd": time_recvd,
        "Source": src
    }
    
    outputs.append(obj)

with open('output.json', 'w') as out_file:
    json.dump(outputs, out_file, indent=4)

# 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