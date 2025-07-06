# GPT Analytics

Analyze your ChatGPT export in seconds. GPT Analytics is a modern full-stack dashboard (Next.js 14 + FastAPI) that surfaces conversation topics, model usage, daily activity, and more — all wrapped in a retro-terminal UI.

## Prerequisites
- Python 3.9 +
- Node.js 18 +

## Quick Start (one-liner)
```bash
git clone https://github.com/yourusername/gpt-analytics
cd gpt-analytics
python scripts/dev.py --serve   # sets up everything & starts both servers
```
Visit **http://localhost:3000**, upload the exported `conversations.json`, and watch the charts appear.

## Manual Start (two terminals)
```bash
# 1 ─ FastAPI backend
python -m venv .venv && source .venv/bin/activate
pip install -r api/requirements.txt
uvicorn api.app.main:app --reload --port 8000

# 2 ─ Next.js frontend
cd web && npm install
npm run dev
```

## License
MIT
