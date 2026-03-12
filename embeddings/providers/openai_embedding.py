from openai import OpenAI
from embeddings.base_embedding import BaseEmbedding
from config.settings import EMBEDDING_MODEL, OPENAI_API_KEY


class OpenAIEmbedding(BaseEmbedding):

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed_documents(self, texts):

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text):

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )

        return response.data[0].embedding