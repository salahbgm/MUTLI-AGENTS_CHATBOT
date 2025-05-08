# src/my_tools/rag_tool.py

from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
#from pinecone import Pinecone
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from utils.config import load_env, get_pinecone_key, get_pinecone_index_name


# Input schema
class RAGInput(BaseModel):
    query: str = Field(..., description="The user's question")

class RAGTool(BaseTool):
    name: str = "rag_tool"
    description: str = "Query the Pinecone index using semantic search"
    args_schema: Type[BaseModel] = RAGInput

    def _run(self, query: str) -> str:
        load_env()

        # Load API keys and config
        api_key = get_pinecone_key()
        index_name = get_pinecone_index_name()

        # Pinecone client (SDK v3)
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Embeddings must match the ones used to index
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        # LangChain wrapper over Pinecone
        #vectorstore = LangChainPinecone(index, embedding.embed_query, text_key="text")
        vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")

        retriever = vectorstore.as_retriever()

        # RAG chain
        llm = ChatOpenAI()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        return qa_chain.run(query)

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported.")
