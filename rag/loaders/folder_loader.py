import os

from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.html_loader import HTMLLoader
from rag.loaders.base_loader import BaseLoader


class FolderLoader(BaseLoader):

    def load(self, folder_path):

        documents = []

        for root, _, files in os.walk(folder_path):

            for file in files:

                file_path = os.path.join(root, file)

                if file.lower().endswith(".pdf"):
                    documents.extend(PDFLoader().load(file_path))

                elif file.lower().endswith((".html", ".htm")):
                    documents.extend(HTMLLoader().load(file_path))

        return documents