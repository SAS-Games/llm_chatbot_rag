import ollama
from llm.base_llm import BaseLLM
from config.settings import LLM_MODEL, LLM_TEMPERATURE;


class OllamaProvider(BaseLLM):

    def generate_response(self, messages):

        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options={
                "temperature" : LLM_TEMPERATURE
            }
        )

        return response["message"]["content"]