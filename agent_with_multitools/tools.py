"""
tools.py

Contient les wrappers pour interagir avec les sources de données externes (Arxiv, Wikipedia, etc.)
"""

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper




# Arxiv
api_wrappers_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrappers_arxiv, description="Query Arxiv for research papers.")
#print(arxiv.name)

# Wikipedia
api_wrappers_wikipedia = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrappers_wikipedia)
#print(wikipedia.name)
