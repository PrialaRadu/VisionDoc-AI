import fitz
import os
import PIL.Image
import io
from docx2pdf import convert
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import json

def extract_images(pdf, page, img_dir):
    img_list = pdf[page].get_images()
    os.makedirs(img_dir, exist_ok=True)
    if img_list:
        for idx, img in enumerate(img_list, start=1):
            data = pdf.extract_image(img[0])
            with PIL.Image.open(io.BytesIO(data.get('image'))) as image:
                image.save(f'{img_dir}/pg{page}-img{idx}.{data.get("ext")}')

def images_data(img_dir):
    results = []
    for img in os.listdir(img_dir):
        results.append({
            'image_path': os.path.join(img_dir, img),
        })
    return results

def extract_text_near_images(pdf_path, expand=50):
    doc = fitz.open(pdf_path)
    results = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 1:  # imagine
                bbox = fitz.Rect(block["bbox"])
                region = fitz.Rect(
                    bbox.x0 - expand,
                    bbox.y0 - expand,
                    bbox.x1 + expand,
                    bbox.y1 + expand
                )
                dict_bbox = {
                    "x0": bbox.x0,
                    "y0": bbox.y0,
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                }
                text = page.get_textbox(region)
                results.append({
                    "page": page_number+1,
                    "image_bbox": dict_bbox,
                    "nearby_text": text.strip()

                })
    doc.close()
    return results

def describe_image(image_path):
    device = torch.device("cpu")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt").to(device)   # datele transformate in tensori PyTorch
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def metadata(image_data, text_data):
    data = []
    for i in range(len(image_data)):
        data.append(image_data[i])
        data[i]['metadata'] = text_data[i]
        data[i]['metadata']['description'] = describe_image(data[i]['image_path'])
        print(f"datele despre imaginea {i} au fost salvate")
    return data

def convert_to_pdf(fpath):
    if fpath.lower().endswith(".docx"):
        convert(fpath, f"data/docxtest.pdf")
        pdf_path = "data/pdftest.pdf"
    else:
        pdf_path = fpath
    return pdf_path

def main():
    filepath = "data/docxtest.docx"
    pdf_path = convert_to_pdf(filepath)
    pdf = fitz.open(pdf_path)
    for page in range(pdf.page_count):
        extract_images(pdf, page, 'img')
    final_data = metadata(images_data('img'), extract_text_near_images(pdf_path))
    for x in final_data:
        print(x)
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()