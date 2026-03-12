import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []


    def add_documents(self, embeddings, docs):

        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

        for doc in docs:
            self.documents.append(doc)


    def search(self, query_embedding, k=4):

        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, k)

        results = []

        for idx in indices[0]:

            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results


    def save(self, index_path, docs_path):

        faiss.write_index(self.index, index_path)

        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)


    def load(self, index_path, docs_path):

        self.index = faiss.read_index(index_path)

        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)