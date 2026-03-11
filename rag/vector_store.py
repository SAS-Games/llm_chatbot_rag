import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(dimension)

        self.documents = []

    def add_documents(self, embeddings, docs):

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)

        self.documents.extend(docs)

    def search(self, query_embedding, k=4):

        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, k)

        results = []

        for idx in indices[0]:
            results.append(self.documents[idx])

        return results