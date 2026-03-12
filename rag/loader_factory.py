import os

from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.folder_loader import FolderLoader

def get_loader(path):

    if os.path.isdir(path):
        return FolderLoader()

    if path.lower().endswith(".pdf"):
        return PDFLoader()

    raise ValueError(f"Unsupported document source: {path}")