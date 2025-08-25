import os
import json
from llama_describe_image import get_description_llama
from extract_from_pdf import extract_images_and_text_pdf
from extract_from_docx import extract_images_and_text_docx


def extract_file(filepath):
    # Verifies if the document is a .pdf or .docx file
    if filepath.endswith(".pdf"):
        return extract_images_and_text_pdf(filepath)
    elif filepath.endswith(".docx"):
        return extract_images_and_text_docx(filepath)
    else:
        raise ValueError(f"unsupported file: {filepath}")

def process_file(filepath):
    results = extract_file(filepath)
    # Generates Gemma3 description for each image
    for item in results:
        desc = get_description_llama(item["image_path"])
        item['description'] = desc

    # Prepares output directory
    filename = filepath.split('/')[-1]
    output_dir = "data/" + filename

    # Writes the info into a json file
    with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# ===============================================================================

if __name__ == '__main__':
    # Extracts all the files that are in the documents/ directory
    files = os.listdir('documents/')
    for file in files:
        process_file("documents/" + file)
