#unused, was seeing if just quantized gemma was a better option
#(it's not)
from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image
# not sure why this is here (the thing above) (and below ig) (did I fuck something up I hope not)
import requests
#guess what didnt work
import torch
from mss import MSS
import time

time.sleep(3)
with MSS() as sct:
    mss = sct.shot()



model_id = "unsloth/gemma-3-4b-it-bnb-4bit"

model = AutoModelForMultimodalLM.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a machine made to output answers to either word or number math problems, outputting the correct answers without reasoning."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": mss},
            {"type": "text", "text": "Provide the answer to the problem in the image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

#ok thanks repo bullshits
# **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
# focusing on a cluster of pink cosmos flowers and a busy bumblebee.
# It has a slightly soft, natural feel, likely captured in daylight.
