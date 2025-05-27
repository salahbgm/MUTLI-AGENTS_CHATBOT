# main.py – Unified Interface for Multi-Tools + DeepResearch
import time

import streamlit as st
import traceback
from langchain_core.messages import HumanMessage, AIMessage

# Internal imports
from workflow import graph as multitask_graph
from tracking import tracker

st.set_page_config(page_title="RFP Assistant", layout="wide")

# Import DeepResearch interface
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DeepResearch_HITL.coordinator import deepresearch_ui

# ─────────────────────────────────────────────
# ⚙️ Session Initialization
# ─────────────────────────────────────────────
st.title("🧠 RFP Assistant")

if "mode" not in st.session_state:
    st.session_state.mode = "multi"  # Default mode

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# 🧭 Agent Mode Switch Bar
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 5, 1])

with col1:
    if st.button("🔄 Reset"):
        tracker.reset()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

with col3:
    if st.button("🔍 DeepResearch" if st.session_state.mode != "deep" else "🧰 Multi-Tools"):
        st.session_state.mode = "deep" if st.session_state.mode != "deep" else "multi"
        st.rerun()

st.markdown(f"### 🎯 Active Agent: `{st.session_state.mode}`")

# ─────────────────────────────────────────────
# 💬 Dynamic Interface Based on Selected Mode
# ─────────────────────────────────────────────
if st.session_state.mode == "multi":
    # User input field
    user_input = st.text_area("💬 Ask your question here:", height=150)

    if st.button("🚀 Run Agent") and user_input.strip() != "":
        try:
            question = user_input.strip()
            st.session_state.conversation.append(HumanMessage(content=question))
            start = time.time()

            result = multitask_graph.invoke({"messages": st.session_state.conversation})
            end = time.time()
            elapsed = round(end - start, 2)

            # 🔍 Try to extract direct output first
            ai_reply = result.get("output") or result.get("answer") or None

            # 🧠 Fallback: retrieve last non-empty AI message
            if not ai_reply:
                for msg in reversed(result.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content.strip():
                        ai_reply = msg.content
                        st.session_state.conversation.append(msg)
                        break

            st.info(f"⏱️ Execution time: {elapsed} seconds")

            if ai_reply:
                st.session_state.history.append({
                    "question": question,
                    "replies": [ai_reply]
                })

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.error(traceback.format_exc())

    # Display conversation history
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown("---")
            st.markdown(f"### ❓ Question #{len(st.session_state.history) - i + 1}")
            st.markdown(f"**{entry['question']}**")

            for j, reply in enumerate(entry['replies'], 1):
                with st.expander(f"✅ Answer {j}", expanded=True):
                    st.markdown(reply)

elif st.session_state.mode == "deep":
    # Render the full DeepResearch interface
    deepresearch_ui()
