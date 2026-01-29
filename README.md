# 🧠 MCP Server & Agentic Workflows
### Smart Research & Business Intelligence Platform  


---

## 🌐 Live Demo
🔗 **Live Website:**  https://research-mcp-frontend-suit-alpha.vercel.app

---
## 📌 Overview

**MCP Server & Agentic Workflows** is a **agent-based, multi-system AI platform** designed to automate **research discovery, intelligence analysis, and business proposal generation** using **agentic workflows**.

This project integrates **four independent AI systems** into a **single orchestrated ecosystem** using a centralized **MCP (Model Context Protocol) Server** and **LangGraph-based agentic pipelines**, executed under a **Scrum of Scrums (SoS)** model as part of **Software Project Management (SPM)**.

---

## 🧩 Integrated Systems

➤ **Volvox** — RAG-based research assistant & knowledge vault  
➤ **Smart Research Answering System** — Simple + deep web search, research paper retrieval  
➤ **Innoscope** — Roadmap generation, feasibility & market analysis  
➤ **Kickstart** — Automated business proposal generation  

Each system operates as an **independent Scrum team**, unified through SoS coordination.

---

## 🧠 Core AI Capabilities

➤ Retrieval-Augmented Generation (RAG)  
➤ Agentic workflows with minimal human input  
➤ Multi-agent orchestration via **LangGraph**  
➤ Centralized context & tool routing using **MCP Server**  
➤ Deep research + web search + internal paper search  
➤ Live agent execution logging via **WebSockets**  
➤ Persistent memory, embeddings, and knowledge storage  

---

## 🔗 MCP Server (System Backbone)

The **MCP Server** acts as the **central nervous system** of the platform.

Responsibilities:
➤ Unified tool exposure across all AI systems  
➤ Standardized input/output schemas  
➤ Context passing between agents and workflows  
➤ Orchestration of cross-system actions  
➤ Decoupling of AI agents from service implementations  

This enabled **contract-first development** and reduced cross-team dependency risks.

---

## 🤖 Agentic Workflows (LangGraph)

All workflows are implemented using **LangGraph**, enabling:
➤ Directed graph-based execution  
➤ Conditional routing  
➤ Stateful memory between nodes  
➤ Autonomous decision-making  

---

## 📊 Workflow 1 — Research → Roadmap → Feasibility → Proposal

**Title:** Business Research to Proposal Generation Pipeline

Flow:
1. Upload research paper to Volvox  
2. Fetch and summarize paper  
3. Generate roadmap via Innoscope  
4. Perform feasibility & market analysis  
5. Auto-generate proposal via Kickstart  
6. Store final artifact in Volvox Knowledge Vault  

---

## 🧪 Workflow 2 — Research Intelligence & Summarization Loop

**Title:** Smart Research Analysis & Summarization

Flow:
1. Query correction & enhancement  
2. Deep search for related work  
3. Fetch internal research papers  
4. Summarize extracted content  
5. Export summary as PDF  
6. Store in Volvox database  

---

## 📈 Workflow 3 — Competitor & Market Intelligence Loop

**Title:** Competitor Research → Market Feasibility → Proposal

Flow:
1. Enter startup/product idea  
2. Perform competitor, funding & patent search  
3. Query expansion & correction  
4. Web search + RAG + data storage  
5. Market trend summarization  
6. Opportunity & feasibility matrix  
7. Auto-generate proposal  
8. Store final output  

---

## 🛠️ Tech Stack

### Backend
➤ Python  
➤ FastAPI  
➤ LangChain  
➤ LangGraph  
➤ MCP Server  
➤ WebSockets  

### Frontend
➤ Next.js  
➤ Real-time agent execution logs  
➤ Unified workflow dashboard  

### Database & Storage
➤ MongoDB  
➤ Vector databases  
➤ PDF & proposal storage  

### AI Architecture
➤ RAG pipelines  
➤ Multi-agent systems  
➤ Autonomous decision graphs  

---

## 📊 Software Project Management (SPM)

### Methodology
➤ **Agile – Scrum of Scrums**

Each system operated as a **separate Scrum team**, coordinated through SoS ceremonies.

---

### Sprint Overview

**Sprint 1 – System Unification & MCP**
➤ Unified deployment  
➤ MCP server implementation  
➤ Workflows 1 & 2  

**Sprint 2 – Advanced Workflows & Dashboard**
➤ Workflows 3 & 4  
➤ Unified workflow dashboard  
➤ Live execution monitoring  

**Sprint 3 – Optimization & Final Integration**
➤ System-wide testing  
➤ Security hardening  
➤ Final demo & class integration  

---

### SPM Artifacts
➤ Product & sprint backlogs  
➤ Sprint plans, reviews & retrospectives  
➤ Kanban boards  
➤ Velocity & burndown charts  
➤ RACI chart  
➤ Risk analysis sheet  
➤ PMI templates  
➤ Dependency & contract-first documentation  

---

## 🎯 Key Learning Outcomes
➤ Large-scale AI system orchestration  
➤ Agentic workflow design  
➤ Multi-team coordination using SoS  
➤ AI + SPM integration  
➤ Production-style system thinking  

---

## 📦 Installation

### Backend Setup
```bash
pip install -r requirements.txt
uvicorn app:mcp_server --reload
```

### Frontend Setup
```bash
npm install
npm run dev
```
