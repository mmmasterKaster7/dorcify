# only pasting ive done here is either from repos or documentation
# vib*coders are the scum of the earth
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch
import mss
import time
from PIL import Image

model_id = "google/paligemma2-10b-mix-448"


time.sleep(3)

#screenshit
with mss.MSS() as sct:
    monitor = sct.monitors[1]

    screen_width = monitor["width"]
    screen_height = monitor["height"]

    mregion = {"top": int(screen_height)*1/3, "left": 0, "width": screen_width, "height": (int(screen_height)*1/3)*2}

    sct_img = sct.grab(mregion)

# I put sct_img as opposed to my own variable name just to be safe bc the documentation always uses that name
# my code only worked when I used that name anyway (despite me very obviously fixing it through something else)
prob_photo = image = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
image = load_image(prob_photo)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

prompt = "OCR en"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False) 
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
