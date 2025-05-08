import os
from dotenv import load_dotenv

def load_env():
    # remonte d’un niveau depuis utils jusqu’à DeepResearch_HITL/
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(dotenv_path=env_path)

def get_openai_key():
    return os.getenv("OPENAI_API_KEY")

def get_tavily_key():
    return os.getenv("TAVILY_API_KEY")
