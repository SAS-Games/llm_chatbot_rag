from config.settings import LLM_PROVIDER

from llm.providers.ollama_provider import OllamaProvider
#from llm.providers.openai_provider import OpenAIProvider


PROVIDERS = {
    "ollama": OllamaProvider,
   # "openai": OpenAIProvider,
}


def get_llm():

    provider_class = PROVIDERS.get(LLM_PROVIDER)

    if not provider_class:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

    return provider_class()