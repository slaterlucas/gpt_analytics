import json
import ijson
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import traceback

def job_status(jid: str, jobs: dict):
    """Get job status"""
    if jid not in jobs:
        return {"error": "Job not found"}
    return jobs[jid]

def extract_messages(file_path: Path):
    """Extract messages from ChatGPT export JSON"""
    messages = []
    try:
        with open(file_path, 'rb') as f:
            # Parse conversations from ChatGPT export
            conversations = ijson.items(f, 'item')
            for conv in conversations:
                if 'mapping' in conv:
                    for node_id, node in conv['mapping'].items():
                        if node and 'message' in node and node['message']:
                            message = node['message']
                            if message.get('author', {}).get('role') == 'user':
                                content = message.get('content', {})
                                if isinstance(content, dict) and 'parts' in content:
                                    text = ' '.join(content['parts'])
                                    if text.strip():
                                        messages.append(text.strip())
    except Exception as e:
        print(f"Error parsing file: {e}")
        # Fallback: try to parse as simple message list
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    messages = [item.get('text', str(item)) for item in data if item]
                elif isinstance(data, dict) and 'messages' in data:
                    messages = [msg.get('text', str(msg)) for msg in data['messages']]
        except:
            pass
    
    return messages

def analyze_topics(messages, job_id: str, jobs: dict):
    """Perform topic modeling on messages"""
    if not messages:
        return {"topics": [], "message": "No messages found"}
    
    # Update progress
    jobs[job_id]["progress"] = 20
    
    # Clean and filter messages
    cleaned_messages = []
    for msg in messages:
        # Remove very short messages
        if len(msg.split()) > 3:
            # Basic cleaning
            msg = re.sub(r'http\S+', '', msg)  # Remove URLs
            msg = re.sub(r'\s+', ' ', msg).strip()  # Normalize whitespace
            if msg:
                cleaned_messages.append(msg)
    
    jobs[job_id]["progress"] = 40
    
    if len(cleaned_messages) < 5:
        return {"topics": [{"topic": "Insufficient Data", "count": len(messages)}]}
    
    try:
        # Use a lightweight sentence transformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        jobs[job_id]["progress"] = 60
        
        # Create topic model
        topic_model = BERTopic(
            embedding_model=sentence_model,
            min_topic_size=max(2, len(cleaned_messages) // 20),  # Adaptive min topic size
            verbose=False
        )
        
        jobs[job_id]["progress"] = 80
        
        # Fit the model
        topics, probs = topic_model.fit_transform(cleaned_messages)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        # Format for frontend
        result_topics = []
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                topic_words = topic_model.get_topic(row['Topic'])
                topic_name = ' '.join([word for word, score in topic_words[:3]])
                result_topics.append({
                    "topic": topic_name.title() if topic_name else f"Topic {row['Topic']}",
                    "count": row['Count']
                })
        
        jobs[job_id]["progress"] = 100
        
        return {"topics": result_topics}
        
    except Exception as e:
        print(f"Topic modeling error: {e}")
        # Fallback: simple keyword extraction
        all_text = ' '.join(cleaned_messages).lower()
        words = re.findall(r'\b\w{4,}\b', all_text)  # Words with 4+ characters
        common_words = Counter(words).most_common(10)
        
        fallback_topics = [
            {"topic": word.title(), "count": count}
            for word, count in common_words
        ]
        
        jobs[job_id]["progress"] = 100
        return {"topics": fallback_topics}

def ingest_stream(file_path: Path, job_id: str, jobs: dict):
    """Background task to process uploaded file"""
    try:
        jobs[job_id]["progress"] = 10
        
        # Extract messages
        messages = extract_messages(file_path)
        
        if not messages:
            jobs[job_id]["error"] = "No messages found in file"
            jobs[job_id]["ready"] = True
            return
        
        # Analyze topics
        result = analyze_topics(messages, job_id, jobs)
        
        # Store results
        jobs[job_id]["result"] = result
        jobs[job_id]["ready"] = True
        jobs[job_id]["message_count"] = len(messages)
        
        # Clean up file
        file_path.unlink()
        
    except Exception as e:
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["ready"] = True
        print(f"Processing error: {traceback.format_exc()}")

def topic_pie(job_id: str, jobs: dict):
    """Get topic analysis results for pie chart"""
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    job = jobs[job_id]
    if not job.get("ready"):
        return {"error": "Job not ready"}
    
    if job.get("error"):
        return {"error": job["error"]}
    
    result = job.get("result", {})
    topics = result.get("topics", [])
    
    # Format for ApexCharts
    return {
        "series": [topic["count"] for topic in topics],
        "labels": [topic["topic"] for topic in topics],
        "message_count": job.get("message_count", 0)
    } 