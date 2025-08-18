import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

def blip2_model_describe(image_path):
    device = torch.device("cpu")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt").to(device)   # datele transformate in tensori PyTorch
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def blip22(image_path):
    device = torch.device("cpu")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs,)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text