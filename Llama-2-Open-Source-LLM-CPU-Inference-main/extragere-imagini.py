import fitz
import os
import PIL.Image
import io

# primeste un pdf cu fitz.open, o pagina de start, si un nume de director
# salveaza toate imaginile gasite in director
def extract_images(pdf, page, img_dir):
    img_list = pdf[page].get_images()
    os.makedirs(img_dir, exist_ok=True)
    if img_list:
        print(page)
        for idx, img in enumerate(img_list, start=1):
            data = pdf.extract_image(img[0])
            with PIL.Image.open(io.BytesIO(data.get('image'))) as image:
                image.save(f'{img_dir}/{page}-{idx}.{data.get("ext")}', mode='wb')

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
                text = page.get_textbox(region)
                results.append({
                    "page": page_number+1,
                    "image_bbox": bbox,
                    "nearby_text": text.strip()
                })
    doc.close()
    return results

def main():
    pdf = fitz.open("data/pdftest.pdf")
    for page in range(pdf.page_count):
        extract_images(pdf, page, 'img')
    print(extract_text_near_images('data/pdftest.pdf'))
    for data in extract_text_near_images('data/pdftest.pdf'):
        print(f"Pe pagina {data['page']} langa poza, se afla textul: {data['nearby_text']}")

if __name__ == '__main__':
    main()