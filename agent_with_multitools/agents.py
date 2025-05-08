# agents.py

from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

# Importation du tracker modifié
from tracking import tracker

from crawl4ai import AsyncWebCrawler
import asyncio

from utils.config import load_env, get_openai_key, get_tavily_key
load_env()
# Charger les variables d'environnement

os.environ["OPENAI_API_KEY"] = get_openai_key()
os.environ["TAVILY_API_KEY"] = get_tavily_key()

# Importation des outils
from tools import wikipedia, arxiv
from my_tools.rag_tool import RAGTool
from langchain_community.tools.tavily_search.tool import TavilySearchResults

# Instanciation
rag_tool = RAGTool()
tavily = TavilySearchResults()

# Définition des inputs via Pydantic et des outils avec marquage [TOOL: ...]

class WikipediaInput(BaseModel):
    query: str

@tool(args_schema=WikipediaInput)
def wikipedia_search(query: str) -> str:
    """Search for information using Wikipedia."""
    result = wikipedia.invoke(query)
    # N'ajouter l'outil que s'il fournit des informations utilisables
    if result and "No good Wikipedia Search Result was found" not in result:
        tracker.add_tool("wikipedia_search", result)
        return result
    else:
        return "No relevant information found on Wikipedia."


class ArxivInput(BaseModel):
    query: str

@tool(args_schema=ArxivInput)
def arxiv_search(query: str) -> str:
    """Search academic papers using Arxiv."""
    result = arxiv.invoke(query)
    # N'ajouter l'outil que s'il fournit des informations utilisables
    if result and len(result.strip()) > 10:  # vérifie que ce n'est pas vide ou presque
        tracker.add_tool("arxiv_search", result)
        return result
    else:
        return "No relevant academic article found on Arxiv."


class TavilyInput(BaseModel):
    query: str

@tool(args_schema=TavilyInput)
def tavily_search(query: str) -> str:
    """Search the web using Tavily."""
    result = tavily.invoke(query)
    
    # Tavily retourne une liste de résultats, donc nous devons la traiter différemment
    if result and isinstance(result, list) and len(result) > 0:
        # Formater les résultats de Tavily en chaîne de caractères
        formatted_results = []
        for item in result:
            if "title" in item and "content" in item:
                formatted_results.append(f"Titre: {item['title']}\nContenu: {item['content']}\n")
            elif isinstance(item, str):
                formatted_results.append(item)
                
        result_text = "\n".join(formatted_results)
        
        if result_text and len(result_text) > 10:
            tracker.add_tool("tavily_search", result)
            return result_text
    
    return "No relevant result found with Tavily."

class RAGInput(BaseModel):
    query: str

@tool(args_schema=RAGInput)
def rag_search(query: str) -> str:
    """Retrieve documents from a Pinecone-powered RAG system."""
    result = rag_tool.invoke(query)
    # N'ajouter l'outil que s'il fournit des informations utilisables
    if result and len(result.strip()) > 10:  # vérifie que ce n'est pas vide ou presque
        tracker.add_tool("rag_search", result)
        return result
    else:
        return "No relevant document found in the knowledge base."


class Crawl4AIInput(BaseModel):
    url: str

@tool(args_schema=Crawl4AIInput)
def crawl4ai_search(url: str) -> str:
    """Crawl a web page and return AI-optimized markdown."""
    async def fetch():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown.fit_markdown or result.markdown.raw_markdown

    try:
        markdown = asyncio.run(fetch())
        if markdown and len(markdown.strip()) > 10:
            tracker.add_tool("crawl4ai_search", markdown)
            return markdown
        else:
            return "Crawling the page did not return any usable content."
    except Exception as e:
        return f"Error while crawling the page: {e}"


# Liste des outils
tools = [crawl4ai_search, rag_search, wikipedia_search, tavily_search, arxiv_search]

tools = [t for t in tools if isinstance(t, BaseTool)]

# LLM et prompt
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a smart AI assistant that uses multiple tools to answer user questions.
     
     IMPORTANT: If a tool doesn't return a relevant or satisfactory result, ALWAYS try another tool.
     
     Here are your tools in order of preference:
     1. tavily_search - use it for up-to-date web information (news, sports events, etc.)
     2. crawl4ai_search - use it when you have a specific URL to analyze
     3. rag_search - use it to search in the internal knowledge base
     4. wikipedia_search - use it for general and factual knowledge
     5. arxiv_search - use it for academic and scientific research
     
     If a tool returns an error message or says no information was found, ALWAYS try another relevant tool.
     
     For questions about recent or upcoming sports events, prioritize using tavily_search.
     
     IMPORTANT ABOUT CHAT HISTORY:
     - You have access to the chat history via the variable {chat_history}.
     - When asked about previous questions or responses, ALWAYS check the chat history.
     - When the user asks about previous conversations or what questions they've asked before, use the chat history to provide accurate answers.
     
     Briefly explain why you're using each tool and summarize the responses clearly and in a structured format.
     Be factual and don't make anything up. If no tool provides results, honestly acknowledge it.
     """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# Création de l'agent
agent = create_tool_calling_agent(llm, tools, prompt)

print("📦 Tool types:")
for t in tools:
    print(f" - {t.name} => {type(t)}")

# Ajout du paramètre chat_history dans l'exécuteur d'agent
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    # Permettre le passage de l'historique des conversations
    allowed_tools=["wikipedia_search", "tavily_search", "arxiv_search", "rag_search", "crawl4ai_search"]
)