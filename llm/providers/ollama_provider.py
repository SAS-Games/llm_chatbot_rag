import ollama
from llm.base_llm import BaseLLM
from config.settings import LLM_MODEL, LLM_TEMPERATURE


class OllamaProvider(BaseLLM):

    def __init__(self):
        self.client = ollama.Client()

    # Normal response
    def generate_response(self, messages):

        response = self.client.chat(
            model=LLM_MODEL,
            messages=messages,
            options={
                "temperature": LLM_TEMPERATURE
            }
        )

        return response["message"]["content"]

    # Streaming response
    def stream_response(self, messages):

        stream = self.client.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            options={
                "temperature": LLM_TEMPERATURE
            }
        )

        for chunk in stream:
            content = chunk["message"].get("content", "")
            if content:
                yield content