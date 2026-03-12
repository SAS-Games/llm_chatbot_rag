from sentence_transformers import SentenceTransformer
from embeddings.base_embedding import BaseEmbedding
from config.settings import EMBEDDING_MODEL


class LocalEmbedding(BaseEmbedding):

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts):

        if not texts:
            return []

        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def embed_query(self, text):

        return self.model.encode(
            [text],
            convert_to_numpy=True
        )[0]