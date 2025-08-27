import os
import json
from utils.llama_describe_image import get_description_llama
from docx_processor import DOCXProcessor
from pdf_processor import PDFProcessor

def extract_file(filepath):
    # Verifies if the document is a .pdf or .docx file
    if filepath.endswith(".pdf"):
        processor = PDFProcessor()
    elif filepath.endswith(".docx"):
        processor = DOCXProcessor()
    else:
        raise ValueError(f"unsupported file: {filepath}")

    return processor.extract_images_and_text(filepath)

def process_file(filepath):
    results = extract_file(filepath)
    # Generates Gemma3 description for each image
    for item in results:
        desc = get_description_llama(item["image_path"])
        item['description'] = desc
        print(f"Fisier: {item["filename"]} si pathul imaginii: {item["image_path"]}")
        print(f"Text langa: {item["nearby_text"]}")
        print(f"Descriere: {desc}")

    # Prepares output directory
    filename = filepath.split('/')[-1]
    output_dir = "data/" + filename

    # Writes the info into a json file
    with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# ===============================================================================

if __name__ == '__main__':
    # Extracts all the files that are in the documents/ directory
    files = os.listdir('../documents/')
    for file in files:
        process_file("../documents/" + file)
