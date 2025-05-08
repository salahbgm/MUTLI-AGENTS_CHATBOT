# 🤖 RFP Assistant

An intelligent multi-agent assistant for research, synthesis, and analysis of complex topics.  
This application features two main modes:

- **🧰 Multi-Tool Agent**: A fast conversational assistant with tool orchestration.
- **🔍 DeepResearch (HITL)**: A deep web research engine with Human-in-the-Loop (HITL) validation.

---

## 🚀 Features

- Unified Streamlit interface with agent switcher
- Workflows orchestrated using **LangGraph**
- Interactive history with human and AI messages
- Cost & token tracking (OpenAI usage)
- Downloadable final research report in Markdown
- Execution timer: global and per-step breakdown
- Modular architecture for easy expansion

---

## 📂 Project Structure

```
RFP_Assistant/
├── agent_with_multitools/
│   ├── main.py <<-- Lanch this main to run the project 
│   ├── workflow.py
│   └── tracking.py
│
├── DeepResearch_HITL/
│   ├── main.py
│   ├── coordinator.py
│   ├── model.py
│   ├── research_agents/
│   └── utils/
│       └── token_tracker.py
│                
└── requirements.txt
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your_username/rfp-assistant.git
cd RFP_Assistant
python -m venv venv
source venv/bin/activate  # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

---

## 🧪 Launching the App

```bash
streamlit run agent_with_multitools/main.py
```

Use the top-right toggle to switch between:

- 🔍 **DeepResearch**
- 🧰 **Multi-Tool Agent**

---

## ⏱️ Execution Time Tracking

- Global session timer shown at the top
- Sidebar expandable menu for:
  - Sub-query generation time
  - LangGraph runtime
  - (coming soon) Final report synthesis time
- Cost estimation in tokens and USD

---

## 🧠 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **LangGraph**
- **LangChain**
- **Tavily API**
- **OpenAI API**

---

## 🔐 Configuration

Create a `.env` file or set environment variables:

```env
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
PINECONE_API_KEY=your-key
PINECONE_ENV=your-key
PINECONE_INDEX_NAME=your-key

```

---

## 📥 Export Options

- Final report `.md` download
- Question/answer history


---

## 📦 Roadmap

- [x] Unified agent interface
- [x] DeepResearch flow with HITL
- [x] Per-step runtime tracking
- [ ] PDF + JSON export
- [ ] Upload & analyze documents

---

## 🪪 License

MIT – Free for personal, academic, and commercial use.
