import ollama
from embeddings.base_embedding import BaseEmbedding
from config.settings import EMBEDDING_MODEL


class OllamaEmbedding(BaseEmbedding):

    def embed_documents(self, texts):

        if not texts:
            return []

        embeddings = []

        for text in texts:

            response = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=text
            )

            embeddings.append(response["embedding"])

        return embeddings


    def embed_query(self, text):

        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )

        return response["embedding"]