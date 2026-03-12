from bs4 import BeautifulSoup
from rag.loaders.base_loader import BaseLoader


class HTMLLoader(BaseLoader):

    def load(self, path):

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        text = soup.get_text(separator=" ")

        return [text]