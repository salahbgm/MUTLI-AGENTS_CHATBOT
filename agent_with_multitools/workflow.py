"""
workflow.py

Contient la logique de workflow entre l'utilisateur, les tools, et l'agent.
"""

from langchain_core.messages.human import HumanMessage
from typing import TypedDict
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage, AIMessage  # Human or AI message
from langgraph.graph.message import add_messages  # Reducers in Langgraph

# Construction du graph LangGraph
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode # Node for the tools
from langgraph.prebuilt import tools_condition # Condition for the tools

from agents import agent_executor, tools   # Importation de l'agent et des outils

from langchain_core.messages import HumanMessage
from schemas import AgentResponse

# Importation du tracker modifié
from tracking import tracker

# Définition de l'état
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_attempts: list[str]  # Pour suivre les tentatives d'outils

# Fonction qui appelle l'agent
def tools_call_llm(state: State):
    from schemas import AgentResponse
    from langchain_core.messages import HumanMessage, AIMessage
    from agents import agent_executor
    from tracking import tracker

    # Réinitialiser le tracker à chaque nouvelle requête
    tracker.reset()
    
    # Récupérer tous les messages pour fournir un contexte à l'agent
    all_messages = state["messages"]
    
    # Obtenir la dernière question de l'utilisateur (le dernier message humain)
    last_human_message = None
    for msg in reversed(all_messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break
            
    if last_human_message is None:
        # Cas improbable: pas de message humain trouvé
        return {
            "messages": [AIMessage(content="Je n'ai pas trouvé de question à répondre.")],
            "tool_attempts": []
        }
    
    human_input = last_human_message.content
    
    # Si c'est une nouvelle requête, réinitialiser les tentatives d'outils
    if "tool_attempts" not in state:
        state["tool_attempts"] = []
    
    # Convertir les messages en format compréhensible pour l'agent
    chat_history = []
    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            chat_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
            chat_history.append({"role": "assistant", "content": msg.content})
    
    # Appeler l'agent avec l'historique complet
    result = agent_executor.invoke({
        "input": human_input,
        "chat_history": chat_history,  # Passer l'historique de conversation
        "tool_attempts": state.get("tool_attempts", [])
    })

    answer = result["output"]
    
    # Récupérer tous les outils utilisés depuis le tracker
    tools_used = tracker.get_tools_string()

    
    final_answer = f"🧠 **Response** : {answer}\n\n🔧 **Tools Used** : `{tools_used}`"

    agent_response = AgentResponse(
        answer=final_answer,
        tool_used=tools_used,
        confidence=1.0
    )

    return {
        "messages": [AIMessage(content=agent_response.answer)],
        "tool_attempts": tracker.get_tools()  # Mettre à jour les tentatives d'outils
    }

# Condition pour vérifier si on doit réessayer avec un autre outil
def should_retry_tool(state: State):
    """
    Checks if the last response indicates that no result was found
    and if there are still tools left to try.
    """
    # If a message contains "no result" or "no information"
    last_message = state["messages"][-1].content.lower()
    tool_attempts = state.get("tool_attempts", [])
    available_tools = [t.name for t in tools]
    
    # Check if there is a failure indication in the message
    failure_indicators = [
        "no result",
        "no information",
        "not found",
        "failure",
        "error while"
    ]

    
    has_failure = any(indicator in last_message for indicator in failure_indicators)
    has_remaining_tools = len(tool_attempts) < len(available_tools)
    
    if has_failure and has_remaining_tools:
        return "retry_tool"
    return "continue"

builder = StateGraph(State)
builder.add_node("tools_call_llm", tools_call_llm)
builder.add_node("tools", ToolNode(tools)) ## Call the tools

# Edges
builder.add_edge(START, "tools_call_llm")
builder.add_conditional_edges("tools_call_llm", tools_condition)
builder.add_edge("tools", "tools_call_llm")
builder.add_edge("tools", END)

graph = builder.compile()

'''
# Invocation
messages = graph.invoke({
    "messages": HumanMessage(content="Hi my name is Salah and I wanted to know who won the last Premier League trophy in 2025")
})


for m in messages["messages"]:
    m.pretty_print()
'''