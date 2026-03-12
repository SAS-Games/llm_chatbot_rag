import os

from rag.loader_factory import get_loader
from rag.text_splitter import split_text
from rag.vector_store import VectorStore

from embeddings.embedding_factory import get_embedding_model
from llm.llm_factory import get_llm


INDEX_FILE = "vector_index.faiss"
DOC_FILE = "vector_docs.pkl"


class RAGService:

    def __init__(self, source_path):

        self.embedding_model = get_embedding_model()
        self.llm = get_llm()

        # If vector index already exists, load it
        if os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):

            print("Loading existing vector index...")

            # dimension must match embedding model
            self.vector_store = VectorStore(384)
            self.vector_store.load(INDEX_FILE, DOC_FILE)

        else:

            print("Building vector index...")

            loader = get_loader(source_path)

            documents = loader.load(source_path)
            print("Documents loaded:", len(documents))

            chunks = []

            for doc in documents:
                chunks.extend(split_text(doc))

            print("Chunks created:", len(chunks))

            if not chunks:
                raise ValueError("No chunks generated from documents.")

            embeddings = self.embedding_model.embed_documents(chunks)

            dimension = len(embeddings[0])

            self.vector_store = VectorStore(dimension)

            self.vector_store.add_documents(embeddings, chunks)

            # Save index for future runs
            self.vector_store.save(INDEX_FILE, DOC_FILE)

            print("Vector index saved.")


    def ask(self, question):

        query_embedding = self.embedding_model.embed_query(question)

        docs = self.vector_store.search(query_embedding, k=3)

        context = "\n".join(docs)

        prompt = f"""
Context:
{context}

Question:
{question}
"""

        messages = [
            {
                "role": "system",
                "content": "Answer questions using ONLY the provided context. If the answer is not in the context, say you don't know."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.llm.generate_response(messages)

        return response