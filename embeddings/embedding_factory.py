from config.settings import EMBEDDING_PROVIDER

from embeddings.providers.ollama_embedding import OllamaEmbedding
from embeddings.providers.local_embedding import LocalEmbedding


PROVIDERS = {
    "ollama": OllamaEmbedding,
    "local": LocalEmbedding
}

_embedding_instance = None


def get_embedding_model():

    global _embedding_instance

    if _embedding_instance is None:
        provider = PROVIDERS.get(EMBEDDING_PROVIDER)
        if not provider:
            raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
        _embedding_instance = provider()

    return _embedding_instance