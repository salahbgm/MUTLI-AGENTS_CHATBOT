from bs4 import BeautifulSoup
import requests
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st

from utils.config import load_env, get_openai_key, get_tavily_key

load_env()

# --- Prompt ---
SEARCH_AGENT_PROMPT = (
    "You are a professional research summarizer specializing in critical content extraction.\n\n"
    "Input:\n"
    "- A URL (already scraped text content).\n"
    "- A title summarizing the topic.\n\n"
    "Task:\n"
    "- Write a highly structured and precise summary.\n"
    "- Summarize ONLY important facts, findings, claims, or statistics.\n"
    "- Remove all irrelevant parts (advertisements, navigation menus, unrelated discussions).\n\n"
    "Formatting:\n"
    "- 2 or 3 **short, dense paragraphs**.\n"
    "- Clear, professional style.\n"
    "- Full sentences. No bullet points.\n"
    "- Neutral tone: NO opinions or speculations.\n\n"
    "Important:\n"
    "- Focus entirely on factual, valuable information.\n"
    "- If the page has almost no useful content, state it explicitly (e.g., 'Minimal relevant content found.').\n"
    "- NEVER invent missing data.\n\n"
    "Output Format Example:\n"
    "\"\"\"\n"
    "Summary:\n"
    "Paragraph 1: [Essential insights and main ideas.]\n\n"
    "Paragraph 2: [Additional critical points, limitations, or supporting data.]\n\n"
    "Optional Paragraph 3: [Only if there are more valuable insights to highlight.]\n"
    "\"\"\""
)

# --- LangChain LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=get_openai_key())

# --- Scraping function ---
def scrape_url(url: str) -> str:
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        return clean_text[:5000] if len(clean_text) > 5000 else clean_text
    except Exception as e:
        return f"Failed to scrape content from {url}: {str(e)}"

# --- Agent runner function ---
async def search_agent(input_text: str) -> str:
    try:
        messages = [
            SystemMessage(content=SEARCH_AGENT_PROMPT),
            HumanMessage(content=input_text)
        ]
        #response = await llm.ainvoke(messages)
        response = await llm.ainvoke(messages, config={"callbacks": [st.session_state.tracker]})

        return response.content.strip()
    except Exception as e:
        raise RuntimeError(f"Search summarization error: {e}")
