import os
from dotenv import load_dotenv

def load_env():
    # 🔁 Charge le fichier .env situé à la racine de agent_with_multitools/
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(dotenv_path=env_path)

def get_openai_key():
    return os.getenv("OPENAI_API_KEY")

def get_tavily_key():
    return os.getenv("TAVILY_API_KEY")

def get_pinecone_key():
    return os.getenv("PINECONE_API_KEY")

def get_pinecone_env():
    return os.getenv("PINECONE_ENV")

def get_pinecone_index_name():
    return os.getenv("PINECONE_INDEX_NAME")
