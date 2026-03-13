from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def generate_response(self, messages):
        pass

    @abstractmethod
    def stream_response(self, messages):
       pass