# Startup War Room 🥊⚖️

A multi-agent AI system where a panel of specialist agents debates your business
idea in real time — researching the market, modeling success probability with a
trained ML model, arguing against it, revising the pitch, and finally delivering
an independent verdict. Built with Google's Agent Development Kit (ADK) and the
Gemini API, deployed live with a persistent case history.

**Live demo:** https://startup-war-room.onrender.com
*(free-tier hosting — may take ~30-60s to wake up if idle)*

---

## What it does

You submit a business idea. Five agents run in sequence:

1. **Market Analyst** — searches the web for market size, competitors, and trends
2. **Financial Modeler** — calls a trained ML model (Random Forest classifier) to
   estimate acquisition probability based on real historical startup data
3. **The Skeptic** ↔ **Synthesizer** — loop back and forth up to 8 rounds; the
   Skeptic attacks the pitch, the Synthesizer revises it, until the Skeptic runs
   out of real objections (or the round cap is hit)
4. **The Judge** — reviews the entire debate independently and delivers a
   structured verdict (`Fund` / `Pass` / `Revisit`, a 1-10 score, and reasoning)
   — deliberately critical, not a rubber stamp

Every case is saved permanently and browsable in a "Case Files" history panel.

## Architecture

```
Browser (courtroom UI, WebSocket)
        │
        ▼
FastAPI backend ── ADK SequentialAgent
        │              ├─ market_analyst  (web_search tool)
        │              ├─ financial_modeler (ML model tool)
        │              ├─ LoopAgent(skeptic ↔ synthesizer, max 8 rounds)
        │              └─ judge
        │
        ▼
Firestore (persistent case history)
```

**Guardrails:** input pitches are screened before any agent runs (blocks
harmful/nonsense submissions); the synthesizer's output is screened for
unrealistic claims ("guaranteed returns", etc.) before reaching the user.

**Resilience:** every external call (web search, Firestore writes, Gemini API)
has a hard timeout with graceful fallback, and a custom rate limiter keeps every
agent under the free-tier API quota.

## The ML model

Trained on a real dataset of ~900 historical US startups (Crunchbase-derived,
1980s–2013), predicting acquisition vs. shutdown from features like funding
rounds, milestones, relationships, and investor participation.

- Compared 4 algorithms via 5-fold cross-validation: Logistic Regression,
  Random Forest, Gradient Boosting, XGBoost
- **Random Forest won** (0.781 AUC vs. 0.76 for the runner-up), tuned via
  Optuna (100 trials)
- Verified real signal via correlation analysis before trusting any feature
  (an earlier candidate dataset was discarded after correlation checks showed
  it was statistically closer to random noise than genuine data)

## Tech stack

| Layer | Tech |
|---|---|
| Agent orchestration | Google ADK (`SequentialAgent`, `LoopAgent`, callbacks) |
| LLM | Gemini (`gemini-3.1-flash-lite`) |
| ML model | scikit-learn `RandomForestClassifier`, tuned with Optuna |
| Backend | FastAPI + WebSockets |
| Database | Google Firestore (Spark/free tier) |
| Frontend | Vanilla HTML/CSS/JS, `marked.js` for markdown rendering |
| Hosting | Render (free tier) |

## Running locally

```bash
pip install -r requirements.txt
python model/train_model.py          # trains and saves the ML model

# .env file:
# GOOGLE_API_KEY=your_gemini_key
# GOOGLE_CLOUD_PROJECT=your_gcp_project
# FIRESTORE_DATABASE_ID=your_firestore_db_name

uvicorn server:app --reload
# open http://127.0.0.1:8000
```

## Project structure

```
startup-war-room/
├── agents/
│   ├── market_analyst.py
│   ├── financial_modeler.py
│   ├── skeptic.py
│   ├── synthesizer.py
│   ├── judge.py
│   └── orchestrator.py        # SequentialAgent + LoopAgent wiring
├── tools/
│   ├── search_tool.py         # web_search with timeout
│   ├── startup_model_tool.py  # ML model wrapped as an ADK tool
│   ├── guardrails.py          # input/output content guardrails
│   └── rate_limiter.py        # keeps calls under free-tier quota
├── model/
│   ├── train_model.py
│   └── startup_success_model.pkl
├── frontend/
│   └── index.html             # courtroom-themed UI
├── server.py                  # FastAPI + WebSocket + Firestore
└── requirements.txt
```
