from rag.document_loader import load_pdf
from rag.text_splitter import split_text
from rag.vector_store import VectorStore

from embeddings.embedding_factory import get_embedding_model
from llm.llm_factory import get_llm


class RAGService:

    def __init__(self, pdf_path):

        self.embedding_model = get_embedding_model()
        self.llm = get_llm()

        text = load_pdf(pdf_path)

        chunks = split_text(text)

        embeddings = self.embedding_model.embed_documents(chunks)

        dimension = len(embeddings[0])

        self.vector_store = VectorStore(dimension)

        self.vector_store.add_documents(embeddings, chunks)

    def ask(self, question):

        query_embedding = self.embedding_model.embed_query(question)

        docs = self.vector_store.search(query_embedding)

        context = "\n".join(docs)

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate_response(messages)

        return response