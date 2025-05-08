import os
import time
from datetime import datetime
from typing import List, TypedDict, Optional, Set

import streamlit as st
from langgraph.graph import StateGraph, END

from DeepResearch_HITL.model import SearchResult
from DeepResearch_HITL.research_agents.query_agent import query_agent, QueryResponse
from DeepResearch_HITL.research_agents.search_agent import search_agent
from DeepResearch_HITL.research_agents.synthesis_agent import synthesis_agent
from DeepResearch_HITL.research_agents.followup_agent import follow_up_decision_agent, FollowUpDecisionResponse
from tavily import TavilyClient


from utils.config import load_env, get_openai_key, get_tavily_key

load_env()

# --- API Client ---
tavily_client = TavilyClient(api_key=get_tavily_key())

# --- State Definition ---
class ResearchState(TypedDict):
    query: str
    subqueries: Optional[List[str]]
    thoughts: Optional[str]
    search_results: List[SearchResult]
    iteration: int
    final_report: Optional[str]
    max_iterations: int  # Ajout du contrôle de profondeur
    processed_queries: Set[str]  # Nouvelle propriété pour suivre les requêtes déjà traitées

# --- LangGraph Nodes ---
async def generate_subqueries_node(state: ResearchState) -> ResearchState:
    query = state["query"]

    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # --- Étape 1 : génération si subqueries n'existent pas en session ---
    if st.session_state.subqueries is None:
        with st.spinner("🔎 Generating sub-queries..."):
            result: QueryResponse = await query_agent(query)

        st.session_state.subqueries = result.queries
        st.session_state.thoughts = result.thoughts
        st.session_state.awaiting_feedback = True
        
        # Ne pas retourner state immédiatement, continuer pour afficher le formulaire

    # --- Étape 2 : interface de feedback utilisateur ---
    if st.session_state.awaiting_feedback:
        form_key = f"subquery_form_{st.session_state.session_id}"
        with st.form(key=form_key, clear_on_submit=True):
            st.subheader("✏️ Validate or Edit Sub-Queries")
            st.markdown(f"**🧠 AI Thoughts:** {st.session_state.thoughts}")

            # Affichage de la profondeur de recherche configurée
            st.info(f"🔍 Research depth: {state.get('max_iterations', 2)} iterations")

            edited_queries = []
            for i, query in enumerate(st.session_state.subqueries):
                edited_query = st.text_input(
                    f"Sub-Query {i + 1}",
                    value=query,
                    key=f"edit_query_{i}_{st.session_state.session_id}"
                )
                edited_queries.append(edited_query)

            feedback = st.text_area(
                "💬 Feedback for improvement (leave blank if satisfied):",
                key=f"feedback_area_{st.session_state.session_id}"
            )

            submitted = st.form_submit_button("✅ Validate Queries")

            if submitted:
                if feedback.strip() == "":
                    st.session_state.awaiting_feedback = False
                    st.session_state.validated_queries = edited_queries
                    st.session_state.step = "continue_graph"  # Signal pour continuer le flux
                    st.rerun()
                else:
                    revised_input = f"Original query: {query}\nUser feedback: {feedback}"
                    with st.spinner("🔄 Regenerating queries based on feedback..."):
                        result = await query_agent(revised_input)
                    st.session_state.subqueries = result.queries
                    st.session_state.thoughts = result.thoughts
                    st.rerun()

        # Arrêter l'exécution pour permettre l'interaction avec le formulaire
        st.stop()

    # --- Étape 3 : feedback validé, mise à jour du state et signal de reprise ---
    # Ce code ne sera exécuté que lorsque awaiting_feedback est False
    state["subqueries"] = st.session_state.validated_queries
    state["thoughts"] = st.session_state.thoughts
    
    # Initialiser processed_queries s'il n'existe pas encore
    if "processed_queries" not in state or state["processed_queries"] is None:
        state["processed_queries"] = set()
        
    return state



def tavily_search(query: str):
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )
        return response.get('results', [])
    except Exception as e:
        st.error(f"❌ Tavily Search Error: {e}")
        return []


async def perform_search_node(state: ResearchState) -> ResearchState:
    st.info("🌐 Starting research on your queries...")

    queries = state.get("subqueries", [])
    search_results = state.get("search_results", [])
    
    # Initialiser processed_queries s'il n'existe pas encore
    if "processed_queries" not in state or state["processed_queries"] is None:
        state["processed_queries"] = set()
    
    processed_queries = state["processed_queries"]

    if not queries:
        st.error("❌ No subqueries found. Cannot continue.")
        raise ValueError("Missing subqueries in state.")

    # Filtrer pour ne traiter que les nouvelles requêtes
    new_queries = [q for q in queries if q not in processed_queries]
    
    if not new_queries:
        st.info("🔄 All queries have already been processed. Moving to next step.")
        # S'il n'y a pas de nouvelles requêtes, on passe à l'étape suivante
        return state

    for query in new_queries:
        with st.spinner(f"🔎 Searching: {query}"):
            # Ajouter la requête à la liste des requêtes traitées
            processed_queries.add(query)
            
            results = tavily_search(query)
            if not results:
                continue

            for result in results:
                input_text = f"Title: {result.get('title', 'No Title')}\nURL: {result.get('url', '')}"
                agent_result = await search_agent(input_text)

                search_results.append(SearchResult(
                    title=result.get('title', 'No Title'),
                    url=result.get('url', ''),
                    summary=agent_result.strip(),
                    query=query  # Sauvegarder la requête associée au résultat
                ))

    state["search_results"] = search_results
    state["processed_queries"] = processed_queries
    return state



async def followup_node(state: ResearchState) -> ResearchState:
    current_iteration = state["iteration"]
    max_iterations = state.get("max_iterations", 2)

    st.info(f"📊 Iteration: {current_iteration + 1}/{max_iterations}")

    if current_iteration >= max_iterations - 1:
        st.info("🔚 Maximum iterations reached. Proceeding to synthesis.")
        state["next"] = "synthesis"
        return state

    findings_text = f"Original Query: {state['query']}\n\nCurrent Findings:\n"
    for i, result in enumerate(state["search_results"], 1):
        findings_text += f"{i}. Title: {result.title}\n   Summary: {result.summary}\n"

    result = await follow_up_decision_agent(findings_text)

    processed_queries = state.get("processed_queries", set())
    unprocessed_queries = [q for q in result.queries if q not in processed_queries]

    if result.should_follow_up and unprocessed_queries:
        st.info(f"🔎 Follow-up Decision: Continue\n\nReason: {result.reasoning}")
        state["subqueries"] = unprocessed_queries
        state["iteration"] += 1
        state["next"] = "perform_search"
    else:
        st.info(f"🛑 Follow-up Decision: Stop OR no new queries\n\nReason: {result.reasoning}")
        state["iteration"] += 1
        state["next"] = "synthesis"

    return state




async def synthesis_node(state: ResearchState) -> ResearchState:
    with st.spinner("📝 Synthesizing final report..."):
        # Créer une liste complète de toutes les requêtes utilisées
        all_queries = list(state.get("processed_queries", set()))
        
        # Organiser les résultats par requête
        query_results = {}
        for result in state["search_results"]:
            query = getattr(result, "query", None)
            if query:
                if query not in query_results:
                    query_results[query] = []
                query_results[query].append(result)
        
        # Préparer le texte d'entrée pour l'agent de synthèse avec une structure basée sur les requêtes
        text = f"# Research Report\n\n**Original Query:** {state['query']}\n\n"
        
        # Ajouter la liste des sous-requêtes utilisées pour la recherche
        text += "## Sub-queries Used for Research:\n"
        for i, query in enumerate(all_queries, 1):
            text += f"{i}. {query}\n"
        
        text += "\n## Research Parameters:\n"
        text += f"- **Iterations completed:** {state['iteration'] + 1}\n"
        text += f"- **Maximum iterations set:** {state.get('max_iterations', 'Not specified')}\n\n"
        
        # Organiser les résultats par requête
        text += "## Search Findings By Query:\n\n"
        for query in all_queries:
            text += f"### Query: {query}\n\n"
            if query in query_results:
                for i, result in enumerate(query_results[query], 1):
                    text += f"#### Result {i}\n"
                    text += f"- **Title:** {result.title}\n"
                    text += f"- **URL:** {result.url}\n"
                    text += f"- **Summary:** {result.summary}\n\n"
            else:
                text += "No specific results for this query.\n\n"

        result = await synthesis_agent(input_text=text)
        state["final_report"] = result

    return state


# --- Fonction pour les arêtes conditionnelles ---
def route_followup(state):
    return state["next"]

# --- Graph Construction ---
graph = StateGraph(ResearchState)

graph.add_node("generate_subqueries", generate_subqueries_node)
graph.add_node("perform_search", perform_search_node)
graph.add_node("followup", followup_node)
graph.add_node("synthesis", synthesis_node)

graph.set_entry_point("generate_subqueries")
graph.add_edge("generate_subqueries", "perform_search")
graph.add_edge("perform_search", "followup")
graph.add_conditional_edges("followup", route_followup, 
                          {
                            "perform_search": "perform_search",
                            "synthesis": "synthesis"
                          })
graph.add_edge("synthesis", END)

app = graph.compile()

# Ce fichier peut maintenant être invoqué avec :
# result = await app.invoke({
#     "query": "your main research question",
#     "search_results": [],
#     "iteration": 0,
#     "max_iterations": 2,  # Nombre d'itérations maximum
#     "processed_queries": set(),  # Initialiser l'ensemble des requêtes traitées
# })


import asyncio

def run_deepresearch(query: str) -> str:
    """
    Fonction de point d’entrée pour lancer le pipeline DeepResearch.
    Appelée depuis main_streamlit.py.
    """

    try:
        state = {
            "query": query,
            "search_results": [],
            "iteration": 0,
            "max_iterations": 2,
            "processed_queries": set(),
        }

        # Lancer l’application LangGraph de façon synchrone
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(app.invoke(state))
        loop.close()

        final_report = result.get("final_report", "Aucun rapport généré.")
        return final_report

    except Exception as e:
        return f"❌ Erreur dans DeepResearch : {e}"



import asyncio
from DeepResearch_HITL.main import main as deep_main  # <- importe ta logique async


def deepresearch_ui():
    """Fonction appelée depuis l'app principale pour afficher l'interface DeepResearch."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(deep_main())
    loop.close()