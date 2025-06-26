# ChatGPT Analytics

A modern, full-stack analytics platform for analyzing your ChatGPT conversation data. Built with Next.js 14 and FastAPI, this project provides topic modeling, conversation insights, and beautiful visualizations.

## ✨ Features

- **One-command setup**: `npm run dev` starts everything
- **Topic Analysis**: Advanced NLP to identify conversation themes
- **Real-time Processing**: Background analysis with progress tracking
- **Beautiful UI**: Modern, responsive design with interactive charts
- **Local & Secure**: All processing happens on your machine
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/chatgpt-analytics
cd gpt-analytics
npm install        # One-time setup (~30 seconds)
npm run dev        # Single command → browser at http://localhost:3000
```

That's it! No Docker, no manual virtual environments, no complex setup.

## 📋 Prerequisites

- **Node.js** 18+ 
- **Python** 3.10+
- **ChatGPT Export File**: Download from ChatGPT settings

## 🏗️ Architecture

```
chatgpt-analytics/
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py        # API endpoints
│   │   └── ingest.py      # Analytics engine
│   └── requirements.txt
├── web/                    # Next.js 14 frontend
│   ├── app/               # App Router pages
│   └── package.json
├── scripts/
│   └── dev.py             # Cross-platform launcher
└── package.json           # Workspace orchestrator
```

## 🔧 How It Works

1. **Upload**: Drag and drop your ChatGPT export JSON
2. **Process**: Advanced NLP extracts topics using BERTopic
3. **Visualize**: Interactive pie charts show conversation themes
4. **Explore**: Dive into your conversation patterns

## 🛠️ Development

The `scripts/dev.py` launcher automatically:
- Creates a Python virtual environment
- Installs all dependencies
- Starts both FastAPI (port 8000) and Next.js (port 3000)
- Enables hot-reload for both services

### Manual Commands

```bash
# Backend only
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uvicorn api.app.main:app --reload --port 8000

# Frontend only
cd web && npm run dev
```

## 📊 API Endpoints

- `POST /upload` - Upload ChatGPT export file
- `GET /status/{job_id}` - Check processing status
- `GET /topics/{job_id}` - Get topic analysis results
- `GET /health` - Health check

## 🔒 Privacy & Security

- All data processing happens locally on your machine
- API keys are stored only in your browser's localStorage
- Uploaded files are automatically deleted after processing
- No data is sent to external services (except OpenAI for embeddings)

## 🎨 Customization

### Adding New Analysis Types

Extend `api/app/ingest.py` with new analysis functions:

```python
def sentiment_analysis(messages, job_id, jobs):
    # Your sentiment analysis logic
    pass
```

### Custom Visualizations

Add new chart types in the dashboard:

```tsx
// In web/app/dashboard/[job_id]/page.tsx
const MyCustomChart = () => {
  // Your chart component
};
```

## 🐛 Troubleshooting

**Port conflicts**: Change ports in `scripts/dev.py`
**Python version**: Ensure Python 3.10+ with `python --version`
**Memory issues**: Reduce file size or increase Node.js memory limit
**CORS errors**: Check that both servers are running on expected ports

## 📈 Performance Tips

- For large files (>100MB), consider chunking the upload
- BERTopic works best with 50+ messages
- Use SSD storage for faster file processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `npm run dev`
5. Submit a pull request

## 📄 License

MIT License - feel free to use for personal or commercial projects.

## 🙏 Acknowledgments

- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Next.js](https://nextjs.org/) for the frontend framework
- [ApexCharts](https://apexcharts.com/) for beautiful visualizations 