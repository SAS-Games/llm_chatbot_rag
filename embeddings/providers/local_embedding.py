from sentence_transformers import SentenceTransformer
from embeddings.base_embedding import BaseEmbedding
from config.settings import EMBEDDING_MODEL


class LocalEmbedding(BaseEmbedding):

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode([text])[0]