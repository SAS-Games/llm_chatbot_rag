from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text, chunk_size=500, overlap=100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_text(text)

    return chunks