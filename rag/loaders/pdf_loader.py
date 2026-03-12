from pypdf import PdfReader
from rag.loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, path):

        reader = PdfReader(path)

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:

                documents.append({
                    "text": text,
                    "source": path,
                    "page": page_number
                })

        return documents