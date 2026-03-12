from pypdf import PdfReader
from rag.loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, path):

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return [text]