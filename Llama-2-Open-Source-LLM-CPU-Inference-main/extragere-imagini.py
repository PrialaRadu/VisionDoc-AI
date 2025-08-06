import fitz
import os
import PIL.Image
import io
from docx2pdf import convert
import json
from llama_describe_image import get_description_llama


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
    for i in range(len(image_data)):
        data.append(image_data[i])
        data[i]['metadata'] = text_data[i]
        data[i]['metadata']['description'] = get_description_llama(data[i]['image_path'])
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
    filepath = "data/PP.pdf"
    pdf_path = convert_to_pdf(filepath)
    pdf = fitz.open(pdf_path)
    for page in range(pdf.page_count):
        extract_images(pdf, page, 'img2')
    final_data = metadata(images_data('img2'), extract_text_near_images(pdf_path))
    for x in final_data:
        print(x)
    with open('data2.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    main()