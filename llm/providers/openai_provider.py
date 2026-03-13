from openai import OpenAI
from llm.base_llm import BaseLLM
from config.settings import LLM_MODEL, OPENAI_API_KEY, LLM_TEMPERATURE, LLM_BASE_URL


class OpenAIProvider(BaseLLM):

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=LLM_BASE_URL
        )

    # Normal response
    def generate_response(self, messages):

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE
        )

        return response.choices[0].message.content


    # Streaming response
    def stream_response(self, messages):

        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta