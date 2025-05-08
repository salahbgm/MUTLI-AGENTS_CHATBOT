from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
import json
import streamlit as st

from utils.config import load_env, get_openai_key, get_tavily_key

load_env()

class QueryResponse(BaseModel):
    thoughts: str
    queries: list[str]

# --- Prompt ---
QUERY_AGENT_PROMPT = """
You are an expert research strategist.

Your task is to:
1. Analyze the user's original research question carefully.
2. Think strategically about the best way to divide and explore the topic comprehensively.
3. Produce **multiple high-quality, focused search queries**, each targeting a different critical aspect of the topic.

Guidelines:
- Generate **at least 3**, but **no more than 6 queries**.
- Cover different angles: causes, impacts, solutions, statistics, examples, future trends, etc.
- Each query must be highly focused, clear, and able to retrieve strong informational content.
- Avoid superficial, redundant, or overly broad queries.

Before writing the queries:
- Write a **detailed thoughts section** explaining how you decomposed the problem and selected the angles.

Respond strictly in the following JSON format:

{
  "thoughts": "Explain your reasoning here...",
  "queries": [
    "First query...",
    "Second query...",
    "Third query...",
    "Fourth query (optional)...",
    "Fifth query (optional)...",
    "Sixth query (optional)..."
  ]
}

Important:
- All queries must be stand-alone and understandable without reading the original question.
- Always prioritize **quality** and **coverage** over quantity.
- DO NOT invent information or go beyond the research scope.
"""

llm = ChatOpenAI(
    model="gpt-4o",  # ou "gpt-3.5-turbo"
    temperature=0.3,
    openai_api_key=get_openai_key(),
)

async def query_agent(input_text: str) -> QueryResponse:
    messages = [
        SystemMessage(content=QUERY_AGENT_PROMPT),
        HumanMessage(content=input_text)
    ]

    #response = await llm.ainvoke(messages)
    response = await llm.ainvoke(messages, config={"callbacks": [st.session_state.tracker]})

    
    content = response.content.strip()

    # 💥 DEBUG : affiche toujours la réponse brute
    print("🔎 Raw LLM output:\n", content)

    try:
        json_start = content.find("{")
        if json_start == -1:
            raise ValueError("No JSON object found in model output.")

        json_data = json.loads(content[json_start:])
        return QueryResponse(**json_data)

    except Exception as e:
        raise RuntimeError(f"❌ Parsing error: {e}\n--- RAW OUTPUT ---\n{content}")
