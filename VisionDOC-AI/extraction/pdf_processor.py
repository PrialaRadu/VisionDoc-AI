from processor.base import DocumentProcessor
from utils.extract_from_pdf import extract_images_and_text_pdf

class PDFProcessor(DocumentProcessor):
    def extract_images_and_text(self, pdf_path, expand=7, zoom=3):
        return extract_images_and_text_pdf(pdf_path, expand, zoom)
