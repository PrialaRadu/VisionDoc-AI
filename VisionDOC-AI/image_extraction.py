import fitz
import os
import PIL.Image
import io
from docx2pdf import convert
import json
from llama_describe_image import get_description_llama
import docx2txt


def save_image(data, img_dir, page, idx):
    with PIL.Image.open(io.BytesIO(data.get('image'))) as image:
        image.save(f'{img_dir}/pg{page}-img{idx}.{data.get("ext")}')

def extract_images(pdf, page, img_dir):
    img_list = pdf[page].get_images()
    os.makedirs(img_dir, exist_ok=True)
    if img_list:
        for idx, img in enumerate(img_list, start=1):
            data = pdf.extract_image(img[0])
            save_image(data, img_dir, page, idx)


def images_data(img_dir):
    results = []
    for img in os.listdir(img_dir):
        results.append({
            'image_path': os.path.join(img_dir, img),
        })
    return results

def extract_text_near_images(doc, expand=50):
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
                    "x0": round(bbox.x0, 2),
                    "y0": round(bbox.y0, 2),
                    "x1": round(bbox.x1, 2),
                    "y1": round(bbox.y1, 2)
                }
                text = page.get_textbox(region)
                results.append({
                    "page": page_number+1,
                    "image_bbox": dict_bbox,
                    "nearby_text": text.strip()

                })
    doc.close()
    return results


def metadata(image_data, text_data):
    data = []

    for i, img in enumerate(image_data):
        item = img.copy()
        if i < len(text_data):
            item['metadata'] = text_data[i]
        else:
            item['metadata'] = {"page": None, "image_bbox": None, "nearby_text": ""}
        data.append(item)

    for item in data:
        desc = get_description_llama(item['image_path'])
        item['metadata']['description'] = desc
        print(desc)

    return data

def convert_to_pdf(fpath):
    if fpath.lower().endswith(".docx"):
        convert(fpath, f"data/docxtest.pdf")
        pdf_path = "data/pdftest.pdf"
    else:
        pdf_path = fpath
    return pdf_path

def extract_docx(fpath, name):
    os.makedirs(name, exist_ok=True)
    docx2txt.process(fpath, name)

def retrieve_images(filepath, name):
    if filepath.lower().endswith(".docx"):
        extract_docx(filepath, name)
    if filepath.lower().endswith(".pdf"):
        pdf = fitz.open(filepath)
        for page in range(pdf.page_count):
            extract_images(pdf, page, name)
        final_data = metadata(images_data(name), extract_text_near_images(pdf))
        for x in final_data:
            print(x)
        with open(f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
    else:
        raise Exception(f"Unsupported file type (no PDF or DOCX): {filepath.lower()}")


if __name__ == '__main__':
    extract_docx("data/Porsche_US Cayenne_Turbo_2006.docx", "Porsche_US")
