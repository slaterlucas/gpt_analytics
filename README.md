# ChatGPT Analytics

A modern, full-stack analytics platform for analyzing your ChatGPT conversation data. Built with Next.js 14 and FastAPI, featuring topic analysis, model usage tracking, daily activity patterns, and beautiful minimalist visualizations.

## ✨ Features

- **One-command setup**: `npm run dev` starts everything
- **Topic Analysis**: Lightweight keyword and phrase extraction from conversations
- **Model Usage Tracking**: See which AI models you use most (GPT-4, o1, etc.)
- **Daily Activity Charts**: Time series visualization of your conversation patterns
- **Real-time Processing**: Background analysis with progress tracking
- **Beautiful Terminal UI**: Minimalist pixelated green design with interactive charts
- **Custom Cursor Effects**: Fun trailing cursor animation
- **Local & Secure**: All processing happens on your machine
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/chatgpt-analytics
cd gpt_analytics
npm install
npm run dev        # Single command → opens browser at http://localhost:3000
```

That's it! The script automatically:
- Installs Python dependencies in a virtual environment
- Installs Node.js dependencies
- Starts both backend (port 8000) and frontend (port 3000)
- Opens your browser to the analytics dashboard

## 📋 Prerequisites

- **Node.js** 18+ 
- **Python** 3.10+
- **ChatGPT Export File**: Download from ChatGPT → Settings → Data Controls → Export

## 🏗️ Architecture

```
gpt_analytics/
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py        # API endpoints
│   │   └── ingest.py      # Lightweight analytics engine
│   └── requirements.txt
├── web/                    # Next.js 14 frontend
│   ├── app/               # App Router pages
│   ├── components/        # UI components + cursor effects
│   └── package.json
├── scripts/
│   └── dev.py             # Cross-platform launcher
└── package.json           # Workspace orchestrator
```

## 🔧 How It Works

1. **Upload**: Select your ChatGPT export JSON file
2. **Process**: Lightweight analysis extracts topics, models, and daily patterns
3. **Visualize**: Interactive charts show:
   - **Topic Distribution**: Conversation themes (donut chart)
   - **Model Usage**: Which AI models you use most (donut chart)
   - **Daily Activity**: Messages per day over time (area chart)
4. **Explore**: Zoom into time periods and dive into your conversation patterns

## 🛠️ Development

The `scripts/dev.py` launcher automatically:
- Creates a Python virtual environment
- Installs all dependencies
- Starts both FastAPI (port 8000) and Next.js (port 3000)
- Enables hot-reload for both services
- Shows a colorful startup banner

### Manual Commands

```bash
# Backend only
cd api
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend only
cd web && npm install && npm run dev
```

## 📊 API Endpoints

- `POST /upload` - Upload ChatGPT export file
- `GET /status/{job_id}` - Check processing status
- `GET /topics/{job_id}` - Get topic analysis results
- `GET /models/{job_id}` - Get model usage statistics
- `GET /daily/{job_id}` - Get daily activity data
- `GET /health` - Health check

## 🔒 Privacy & Security

- All data processing happens locally on your machine
- OpenAI API key is optional (only for advanced features)
- Uploaded files are automatically deleted after processing
- No data is sent to external services
- Theme preferences stored locally

## 🎨 Customization

### Adding New Analysis Types

Extend `api/app/ingest.py` with new analysis functions:

```python
def weekly_patterns(daily_messages, job_id, jobs):
    # Your weekly pattern analysis logic
    pass
```

### Custom Visualizations

Add new chart types in the main page:

```tsx
// In web/app/page.tsx
const createWeeklyChart = (data: any) => {
  // Your chart component
};
```

## 🐛 Troubleshooting

**Port conflicts**: Change ports in `scripts/dev.py`
**Python version**: Ensure Python 3.10+ with `python --version`
**File upload issues**: Ensure you're uploading the conversations.json file from ChatGPT export
**CORS errors**: Check that both servers are running on expected ports (3000 & 8000)
**Chart not loading**: Clear browser cache and ensure JavaScript is enabled

## 📈 Performance Tips

- Works great with any size conversation file
- Lightweight processing - no heavy ML dependencies
- Fast startup time (~10 seconds)
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

- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Next.js](https://nextjs.org/) for the frontend framework
- [ApexCharts](https://apexcharts.com/) for beautiful visualizations
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [Lucide Icons](https://lucide.dev/) for iconography 