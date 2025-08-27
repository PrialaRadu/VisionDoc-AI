from processor.base import DocumentProcessor
from utils.extract_from_docx import extract_images_and_text_docx

class DOCXProcessor(DocumentProcessor):
    def extract_images_and_text(self, docx_path, expand=7, zoom=3):
        return extract_images_and_text_docx(docx_path)

