import os
import fitz

#========== PDF FUNCTIONS ==========
def create_output_directory(pdf_path):
    # Prepares output directory
    filename = pdf_path.split('/')[-1]
    output_dir = "data/" + filename
    # Creates the output directory
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_nearby_text_from_block(page, block, expand):
    # If the block is an image
    if block["type"] != 1:
        return None

    # Creates a bbox that represents the image
    bbox = fitz.Rect(block["bbox"])
    # Expands the regions based on the expand value
    region = bbox + (-expand, -expand, expand, expand)
    # Retrieves the relevant text in the region
    nearby_text = page.get_textbox(region).strip()
    return nearby_text

def get_results_from_blocks(doc, expand, zoom, output_dir):
    results = []
    # Iterates every document page
    for page_number in range(len(doc)):
        page = doc[page_number]

        # Retrieves every block from the page (text, image, etc.)
        blocks = page.get_text("dict")["blocks"]
        img_index = 1

        # Iterates every block
        for block in blocks:
            # If the block is an image
            if block["type"] != 1:
                continue

            # Creates a bbox that represents the image
            bbox = fitz.Rect(block["bbox"])
            # Retrieves the relevant text in the region
            nearby_text = get_nearby_text_from_block(page, block, expand)

            try:
                # Retrieves the image in a pix object using matrix
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, clip=bbox)

                # Saves the image with a relevant name
                image_name = f"page{page_number + 1}_img{img_index}.png"
                image_path = os.path.join(output_dir, image_name)
                pix.save(image_path)

                # Appends the relevant attributes to the result list, as a dict object
                results.append({
                    "image_path": image_path,
                    "nearby_text": nearby_text,
                    "position": {
                        "x0": bbox.x0,
                        "y0": bbox.y0,
                        "x1": bbox.x1,
                        "y1": bbox.y1,
                    },
                    "page_number": page_number + 1
                })

                img_index += 1

            except Exception as e:
                print(f"[Eroare] Pagina {page_number + 1}, imaginea {img_index}: {e}")

    return results


def extract_images_and_text_pdf(pdf_path, expand=7, zoom=3):
    """
    Extracts images and relevant text from a PDF file using PyMuPdf library.
    param: pdf_path (the path to the PDF file)
    expand: the distance between the image and the relevant text near it
        E.g.: if the value is set to 7, nearby_text will collect every text that is 7 pixels near the image.
    zoom: specify zoom or shear values (float) and create a zoom or shear matrix, respectively.

    returns: a list of dict objects, containing information (image_path, nearby_text, position, page_number) for each image
    """
    output_dir = create_output_directory(pdf_path)

    # Opens the document
    doc = fitz.open(pdf_path)

    # Iterates every document page
    results = get_results_from_blocks(doc, expand, zoom, output_dir)

    # Closes the document
    doc.close()
    return results