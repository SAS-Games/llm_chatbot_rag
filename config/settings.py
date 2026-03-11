from dotenv import load_dotenv
import os

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")