import ollama
from embeddings.base_embedding import BaseEmbedding
from config.settings import EMBEDDING_MODEL


class OllamaEmbedding(BaseEmbedding):

    def embed_documents(self, texts):

        if not texts:
            return []

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts
        )

        return response["embeddings"]


    def embed_query(self, text):

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=[text]
        )

        return response["embeddings"][0]