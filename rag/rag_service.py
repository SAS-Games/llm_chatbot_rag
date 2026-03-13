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

            if isinstance(doc, dict):

                text = doc.get("text", "")
                source = doc.get("source", source_path)
                page = doc.get("page", None)

            elif hasattr(doc, "page_content"):

                text = doc.page_content
                source = doc.metadata.get("source", source_path)
                page = doc.metadata.get("page", None)

            else:

                text = str(doc)
                source = source_path
                page = None

        # clean text
            if not text:
                continue

            text = text.strip()

            if len(text) < 10:
                continue

            for chunk in split_text(text):

                if chunk.strip():

                    chunks.append({
                    "text": chunk,
                    "source": source,
                    "page": page
                })

        print("Documents loaded:", len(documents))
        print("Chunks created:", len(chunks))
        
        if not chunks:
            raise ValueError("No chunks generated from documents")

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        dimension = len(embeddings[0])

        self.vector_store = VectorStore(dimension)

        self.vector_store.add_documents(embeddings, chunks)

        self.vector_store.save(index_file, doc_file)

        print("Vector index saved")


    def ask(self, question, stream=False):

        query_embedding = self.embedding_model.embed_query(question)

        docs = self.vector_store.search(query_embedding, k=8)

        context = "\n".join([d["text"] for d in docs[:4]])

        sources = []

        for d in docs[:4]:

            src = os.path.basename(d["source"])

            if d["page"]:
                src = f"{src} (page {d['page']})"

            sources.append(src)

        prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions using the provided context. "
                    "Rules: "
                    "Use the context to answer questions. "
                    "If the user explicitly asks for sources, references, links, or where the information came from, include the sources. Otherwise do not mention sources"
                    "If the answer is not in the context, say it was not found in the knowledge base."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Streaming mode
        if stream:
            for token in self.llm.stream_response(messages):
                yield token, sources

        # Normal mode
        else:
            response = self.llm.generate_response(messages)
            return response, sources            