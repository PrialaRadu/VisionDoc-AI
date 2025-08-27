from abc import ABC, abstractmethod


class DocumentProcessor(ABC):
    @abstractmethod
    def extract_images_and_text(self, file_path, expand=7, zoom=3):
        pass
