from bs4 import BeautifulSoup
from rag.loaders.base_loader import BaseLoader


class HTMLLoader(BaseLoader):

    def load(self, file_path):

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        text = soup.get_text(separator=" ")

        return [{
            "text": text,
            "source": file_path,
            "page": None
        }]