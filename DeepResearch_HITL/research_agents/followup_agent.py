from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import List
import json
import re
import streamlit as st

from utils.config import load_env, get_openai_key, get_tavily_key

load_env()

# --- Modèle de sortie typé ---
class FollowUpDecisionResponse(BaseModel):
    should_follow_up: bool
    reasoning: str
    queries: List[str]

# --- Prompt système ---
FOLLOW_UP_DECISION_PROMPT = (
    "You are an expert research reviewer.\n\n"
    "You will receive:\n"
    "- The original user query.\n"
    "- Summaries of the current research findings.\n\n"
    "Your tasks:\n"
    "1. Analyze whether the research findings sufficiently and accurately answer the original query.\n"
    "2. If important aspects are missing, propose 2-3 specific follow-up search queries to fill the gaps.\n\n"
    "Decision Rules:\n"
    "- If the original question is **simple and factual** (e.g., 'What is the height of Mount Everest?') and the facts are already present, **do NOT propose follow-up queries**.\n"
    "- If the original question is **complex** (e.g., analysis, causes, comparisons, advantages/disadvantages) and important points are missing, **you MUST propose follow-up queries**.\n"
    "- If findings seem sufficient, be confident and conclude no further research is needed.\n\n"
    "Formatting of your answer (mandatory):\n"
    "```json\n"
    "{\n"
    "  \"should_follow_up\": true or false,\n"
    "  \"reasoning\": \"Detailed explanation for your decision\",\n"
    "  \"queries\": [\n"
    "    \"Follow-up query 1\",\n"
    "    \"Follow-up query 2\",\n"
    "    \"(Optional) Follow-up query 3\"\n"
    "  ]\n"
    "}\n"
    "```\n\n"
    "Additional Notes:\n"
    "- Be analytical and precise.\n"
    "- Never propose follow-ups if they are not necessary.\n"
    "- Focus on covering ALL relevant aspects of the original question completely."
)

# --- Initialisation LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=get_openai_key())

# --- Fonction agent ---
async def follow_up_decision_agent(input_text: str) -> FollowUpDecisionResponse:
    try:
        messages = [
            SystemMessage(content=FOLLOW_UP_DECISION_PROMPT),
            HumanMessage(content=input_text)
        ]
        #response = await llm.ainvoke(messages)
        response = await llm.ainvoke(messages, config={"callbacks": [st.session_state.tracker]})

        content = response.content.strip()

        # Enhanced JSON extraction
        # First, try to extract JSON from markdown code blocks
        json_pattern = r"```(?:json)?(.*?)```"
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        if json_matches:
            # Use the first JSON block found
            potential_json = json_matches[0].strip()
            try:
                parsed = json.loads(potential_json)
                return FollowUpDecisionResponse(**parsed)
            except json.JSONDecodeError:
                pass  # If this fails, continue to other methods
        
        # Try to find JSON directly in the content
        try:
            # Try to find the start of a JSON object
            json_start = content.find("{")
            if json_start != -1:
                # Try to find the end of the JSON object
                json_content = content[json_start:]
                # Parse the JSON
                parsed = json.loads(json_content)
                return FollowUpDecisionResponse(**parsed)
        except json.JSONDecodeError:
            pass
        
        # If all automatic methods fail, use a more aggressive approach
        # Extract what looks like a JSON object
        matches = re.search(r'({.*})', content, re.DOTALL)
        if matches:
            try:
                json_str = matches.group(1)
                parsed = json.loads(json_str)
                return FollowUpDecisionResponse(**parsed)
            except json.JSONDecodeError:
                pass
                
        # If we still couldn't parse the JSON, raise an error with diagnostic info
        raise ValueError(f"Failed to extract valid JSON from content:\n{content}")

    except Exception as e:
        raise RuntimeError(f"Error parsing follow-up response: {e}\nRaw output:\n{content}")