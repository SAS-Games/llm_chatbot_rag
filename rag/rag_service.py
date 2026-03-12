import os

from rag.loader_factory import get_loader
from rag.text_splitter import split_text
from rag.vector_store import VectorStore

from embeddings.embedding_factory import get_embedding_model
from llm.llm_factory import get_llm


class RAGService:

    def __init__(self, source_path, source_hash):

        self.embedding_model = get_embedding_model()
        self.llm = get_llm()

        os.makedirs("indexes", exist_ok=True)

        index_file = f"indexes/{source_hash}.faiss"
        doc_file = f"indexes/{source_hash}.pkl"

        # Load existing index
        if os.path.exists(index_file) and os.path.exists(doc_file):

            print("Loading existing vector index")

            self.vector_store = VectorStore(384)
            self.vector_store.load(index_file, doc_file)

            return

        print("Building vector index")

        loader = get_loader(source_path)

        documents = loader.load(source_path)

        print("Documents loaded:", len(documents))

        chunks = []

        for doc in documents:

            if hasattr(doc, "page_content"):
                text = doc.page_content
            else:
                text = doc

            chunks.extend(split_text(text))

        print("Chunks created:", len(chunks))

        if not chunks:
            raise ValueError("No chunks generated from documents")

        embeddings = self.embedding_model.embed_documents(chunks)

        dimension = len(embeddings[0])

        self.vector_store = VectorStore(dimension)

        self.vector_store.add_documents(embeddings, chunks)

        self.vector_store.save(index_file, doc_file)

        print("Vector index saved")


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