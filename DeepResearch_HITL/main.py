import streamlit as st
from datetime import datetime
import asyncio

from DeepResearch_HITL.coordinator import app, generate_subqueries_node
from DeepResearch_HITL.utils.token_tracker import TokenCostTracker


# ─────────────────────────────────────────────
# INIT SESSION WRAPPER
# ─────────────────────────────────────────────
def init_deepresearch_session():
    if "step" not in st.session_state:
        st.session_state.step = "input_query"

    if "query" not in st.session_state:
        st.session_state.query = ""

    if "intermediate_state" not in st.session_state:
        st.session_state.intermediate_state = None

    if "result" not in st.session_state:
        st.session_state.result = None

    if "subqueries" not in st.session_state:
        st.session_state.subqueries = None

    if "awaiting_feedback" not in st.session_state:
        st.session_state.awaiting_feedback = False

    if "max_iterations" not in st.session_state:
        st.session_state.max_iterations = 2

    if "tracker" not in st.session_state:
        st.session_state.tracker = TokenCostTracker()

    if "timings" not in st.session_state:
        st.session_state.timings = {}

    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now()


# ─────────────────────────────────────────────
# MAIN INTERFACE
# ─────────────────────────────────────────────
async def main():
    init_deepresearch_session()

    st.markdown(f"⏱️ **Temps écoulé :** `{round((datetime.now() - st.session_state.start_time).total_seconds(), 2)}s`")

    # --- ÉTAPE 1 : INPUT DE LA QUESTION ---
    if st.session_state.step == "input_query":
        query = st.text_area("📥 Enter your main research question:", height=150)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.max_iterations = st.slider(
                "🔍 Research depth (maximum number of follow-up iterations):",
                min_value=1,
                max_value=5,
                value=2,
                help="Higher values will result in more thorough research but will take longer."
            )

        with col2:
            if st.button("Start Research", use_container_width=True):
                st.session_state.query = query.strip()
                st.session_state.step = "generate_subqueries"
                st.rerun()

    # --- ÉTAPE 2 : GÉNÉRATION DES SUBQUERIES ---
    elif st.session_state.step == "generate_subqueries":
        with st.spinner("🔎 Generating research sub-queries..."):
            start_step = datetime.now()
            result = await generate_subqueries_node({
                "query": st.session_state.query,
                "subqueries": None,
                "thoughts": None,
                "search_results": [],
                "iteration": 0,
                "final_report": None,
                "max_iterations": st.session_state.max_iterations,
                "processed_queries": set()
            })
            st.session_state.intermediate_state = result
            st.session_state.timings["generate_subqueries"] = (datetime.now() - start_step).total_seconds()
            st.rerun()

    # --- ÉTAPE 3 : ATTENTE DE LA VALIDATION DE L'UTILISATEUR ---
    elif st.session_state.step == "wait_user_feedback":
        pass

    # --- ÉTAPE 4 : CONTINUER LE GRAPHE APRÈS VALIDATION ---
    elif st.session_state.step == "continue_graph":
        with st.spinner("🚀 Launching research on validated queries..."):
            initial_state = {
                "query": st.session_state.query,
                "subqueries": st.session_state.validated_queries,
                "thoughts": st.session_state.thoughts,
                "search_results": [],
                "iteration": 0,
                "final_report": None,
                "max_iterations": st.session_state.max_iterations,
                "processed_queries": set()
            }
            start_step = datetime.now()
            result = await app.ainvoke(initial_state, config={"callbacks": [st.session_state.tracker]})
            st.session_state.result = result
            st.session_state.timings["run_graph"] = (datetime.now() - start_step).total_seconds()
            st.session_state.step = "display_result"
            st.rerun()

    # --- ÉTAPE 5 : AFFICHAGE DU RAPPORT FINAL ---
    elif st.session_state.step == "display_result":
        result = st.session_state.result
        final_report = result.get("final_report", "")

        st.success("🎉 Research complete!")
        st.subheader("📝 Final Report")
        st.markdown(final_report)

        st.sidebar.subheader("📊 Research Statistics")
        st.sidebar.info(f"📋 Total queries: {len(result.get('processed_queries', set()))}")
        st.sidebar.info(f"🔄 Iterations completed: {result.get('iteration', 0) + 1}")
        st.sidebar.info(f"🔍 Max depth set: {result.get('max_iterations', 'N/A')}")

        # Statistiques de coût
        tokens, cost = st.session_state.tracker.get_report()
        st.sidebar.info(f"💰 Tokens used: {tokens}")
        st.sidebar.info(f"💵 Estimated cost: ${cost}")

        # ⏱️ Temps d'exécution global
        total_duration = (datetime.now() - st.session_state.start_time).total_seconds()
        st.sidebar.info(f"⏱️ Total runtime: {round(total_duration, 2)} seconds")

        # ⏳ Timings détaillés
        with st.sidebar.expander("⏳ Timings per step"):
            for step_name, duration in st.session_state.timings.items():
                st.markdown(f"**{step_name}**: {round(duration, 2)} seconds")

        # Téléchargement du rapport
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        st.download_button("📥 Download report", final_report, file_name=filename, mime="text/markdown")

        if st.button("🔁 Start Over"):
            tracker = st.session_state.tracker
            st.session_state.clear()
            st.session_state.tracker = tracker
            tracker.reset()
            st.rerun()
