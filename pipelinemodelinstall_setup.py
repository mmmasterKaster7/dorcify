# I'd reckon this would be somewhat more user-friendly for downloading the model initially
from transformers import pipeline
from huggingface_hub import login
login("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")

pee = ""

pipeline = pipeline(
    task="image-text-to-text",
    model="google/paligemma2-10b-mix-448",
    device=0,
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="What is in this image?"
)