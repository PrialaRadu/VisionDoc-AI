import os
import re
from docx2python import docx2python
from docx2txt import docx2txt

#========== DOCX FUNCTIONS ==========

def save_docx_images(docx_path):
    # Prepares output directory
    filename = docx_path.split('/')[-1]
    output_dir = "data/" + filename + "/images"
    os.makedirs(output_dir, exist_ok=True)
    # Saves images from docx into desired directory
    docx2txt.process(docx_path, output_dir)


def retrieve_first_sentence_after_img(block):
    image_match = re.match(r'(\d+\.\w+)', block)
    if image_match:
        # image_name = f"image{image_match.group(1)}"
        # Searches all sentences
        span_match = re.search(r'<span[^>]*>(.*?)</span>', block)
        # Retrieves the first sentence found
        first_span_text = span_match.group(1).strip().split('.')[0]
        return first_span_text
    else:
        return None


def extract_images_and_text_docx(docx_path):
    """
    Extracts images and relevant text from a DOCX file using docx2python library.
    param: docx_path (the path to the DOCX file)

    returns: a list of dict objects, containing information (image_path, nearby_text) for each image
    """
    directory = "data/" + docx_path.split('/')[-1] + "/images"

    # Saves images
    save_docx_images(docx_path)

    # Reads HTML info from docx
    with docx2python(docx_path, html=True) as docx_content:
        content = docx_content.text

    # Splits info for each photo
    blocks = re.split(r'----media/image', content)

    # Searches sentences that are found after each image
    results = []
    for block in blocks[1:]:
        nearby_text = retrieve_first_sentence_after_img(block)
        match = re.match(r'(\d+\.\w+)', block)
        if match:
            image_name = f"image{match.group(1)}"
        else:
            image_name = "image_unknown"
        # Appends the relevant attributes to the result list, as a dict object
        results.append({
            "image_path": directory + image_name,
            "nearby_text": nearby_text
        })

    return results
