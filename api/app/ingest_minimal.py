import json
import re
from pathlib import Path
from collections import Counter
import traceback

def job_status(jid: str, jobs: dict):
    """Get job status"""
    if jid not in jobs:
        return {"error": "Job not found"}
    return jobs[jid]

def extract_messages_and_models(file_path: Path):
    """Extract messages and model usage from ChatGPT export JSON - COMPREHENSIVE parsing"""
    messages = []
    model_usage = Counter()
    conversation_count = 0
    total_messages = 0
    content_types = Counter()
    conversation_titles = []  # NEW: Store titles for topic analysis
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different ChatGPT export formats
        if isinstance(data, list):
            # Array of conversations
            conversations = data
        elif isinstance(data, dict):
            # Single conversation or wrapper object
            conversations = [data] if 'mapping' in data else data.get('conversations', [data])
        else:
            conversations = []
            
        for conv in conversations:
            if not isinstance(conv, dict) or 'mapping' not in conv:
                continue
                
            conversation_count += 1
            
            # NEW: Extract conversation title for topic analysis
            title = conv.get('title', '').strip()
            if title and title != 'Untitled' and len(title) > 3:
                conversation_titles.append(title)
            
            for node_id, node in conv['mapping'].items():
                if not node or 'message' not in node or not node['message']:
                    continue
                    
                message = node['message']
                author = message.get('author', {})
                role = author.get('role')
                content = message.get('content', {})
                
                # Skip system messages that are empty or hidden
                metadata = message.get('metadata', {})
                if metadata.get('is_visually_hidden_from_conversation'):
                    continue
                
                total_messages += 1
                
                # Extract ALL content types and text
                extracted_text = extract_all_content(content, content_types)
                
                # Add to messages for topic analysis if we got meaningful text
                if extracted_text and len(extracted_text.strip()) > 5:  # Less restrictive
                    # Prefix with role for context
                    prefixed_text = f"[{role}] {extracted_text}" if role else extracted_text
                    messages.append(prefixed_text)
                
                # Extract model usage from assistant messages
                if role == 'assistant':
                    model_slug = metadata.get('model_slug') or metadata.get('default_model_slug')
                    if model_slug:
                        cleaned_model = clean_model_name(model_slug)
                        model_usage[cleaned_model] += 1
                        
    except Exception as e:
        print(f"Error parsing file: {e}")
        traceback.print_exc()
    
    return messages, model_usage, conversation_count, total_messages, content_types, conversation_titles

def extract_all_content(content, content_types):
    """Extract text from all possible content types"""
    if not content:
        return ""
    
    extracted_parts = []
    
    # Handle different content structures
    if isinstance(content, str):
        return content.strip()
    
    if not isinstance(content, dict):
        return str(content).strip()
    
    # Track content type
    content_type = content.get('content_type', 'unknown')
    content_types[content_type] += 1
    
    # Extract based on content type
    if content_type == 'text':
        # Standard text content
        parts = content.get('parts', [])
        if isinstance(parts, list):
            extracted_parts.extend([str(part) for part in parts if part])
        elif parts:
            extracted_parts.append(str(parts))
            
    elif content_type == 'thoughts':
        # Reasoning/thoughts content
        thoughts = content.get('thoughts', [])
        if isinstance(thoughts, list):
            for thought in thoughts:
                if isinstance(thought, dict):
                    summary = thought.get('summary', '')
                    content_text = thought.get('content', '')
                    if summary:
                        extracted_parts.append(f"[Thought Summary: {summary}]")
                    if content_text:
                        extracted_parts.append(f"[Thought: {content_text}]")
                elif thought:
                    extracted_parts.append(f"[Thought: {str(thought)}]")
                    
    elif content_type == 'code':
        # Code content
        code_text = content.get('text', '')
        language = content.get('language', 'unknown')
        if code_text:
            extracted_parts.append(f"[Code ({language}): {code_text[:200]}...]")  # Truncate long code
            
    elif content_type == 'multimodal_text':
        # Multimodal content (text + images, etc.)
        parts = content.get('parts', [])
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    if part.get('content_type') == 'text':
                        text = part.get('text', '')
                        if text:
                            extracted_parts.append(text)
                    elif part.get('content_type') == 'image_asset_pointer':
                        extracted_parts.append("[Image attachment]")
                elif part:
                    extracted_parts.append(str(part))
    else:
        # Fallback: try to extract any text we can find
        if 'parts' in content:
            parts = content['parts']
            if isinstance(parts, list):
                extracted_parts.extend([str(part) for part in parts if part])
        elif 'text' in content:
            extracted_parts.append(str(content['text']))
        else:
            # Last resort: convert whole content to string
            text_content = str(content)
            if len(text_content) < 500:  # Don't include massive JSON dumps
                extracted_parts.append(text_content)
    
    return ' '.join(extracted_parts).strip()

def clean_model_name(model_slug):
    """Clean and normalize model names for better display"""
    if not model_slug:
        return "Unknown"
    
    # Common model mappings
    model_mappings = {
        'gpt-4': 'GPT-4',
        'gpt-4-turbo': 'GPT-4 Turbo',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o Mini',
        'gpt-3.5-turbo': 'GPT-3.5 Turbo',
        'o1': 'o1',
        'o1-preview': 'o1 Preview',
        'o1-mini': 'o1 Mini',
        'o3': 'o3',
        'o3-mini': 'o3 Mini',
        'claude': 'Claude',
        'text-davinci': 'GPT-3 Davinci'
    }
    
    # Try exact match first
    if model_slug in model_mappings:
        return model_mappings[model_slug]
    
    # Try partial matches
    for key, value in model_mappings.items():
        if key in model_slug.lower():
            return value
    
    # Default: capitalize and clean
    return model_slug.replace('-', ' ').title()

def analyze_model_usage(model_usage):
    """Analyze model usage and return percentage breakdown"""
    if not model_usage:
        return []
    
    total_uses = sum(model_usage.values())
    model_stats = []
    
    for model, count in model_usage.most_common():
        percentage = (count / total_uses) * 100
        model_stats.append({
            "model": model,
            "count": count,
            "percentage": round(percentage, 1)
        })
    
    return model_stats

def simple_topic_analysis(messages, job_id: str, jobs: dict, conversation_titles=None):
    """Enhanced topic analysis that handles all content types - TITLE-BASED for efficiency"""
    
    # NEW: Use conversation titles instead of full messages for topic analysis
    if conversation_titles and len(conversation_titles) > 0:
        # Use conversation titles directly for topic analysis
        all_text = conversation_titles
        print(f"DEBUG: Using {len(all_text)} conversation titles for topic analysis")
        combined_text = ' '.join(all_text).lower()
        mode_suffix = " (Title-Based)"
    else:
        # Fallback to message content if no titles available
        if not messages:
            return {"topics": [], "message": "No messages or titles found"}
        
        jobs[job_id]["progress"] = 20
        
        # Combine all messages, but clean them better
        all_text = []
        for msg in messages:
            # Remove role prefixes for analysis but keep the content
            cleaned_msg = re.sub(r'^\[.*?\]\s*', '', msg)
            # Remove special markers but keep the content
            cleaned_msg = re.sub(r'\[.*?:\s*', '', cleaned_msg)
            cleaned_msg = re.sub(r'\]', '', cleaned_msg)
            if len(cleaned_msg.strip()) > 10:  # Keep longer messages
                all_text.append(cleaned_msg.lower())
        
        combined_text = ' '.join(all_text)
        print(f"DEBUG: Using {len(all_text)} message documents for topic analysis")
        mode_suffix = " (Message-Based)"
    
    jobs[job_id]["progress"] = 20
    
    # For titles, use simpler stop words since titles are already concise
    if conversation_titles:
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'how', 'what', 'why', 'when', 
            'where', 'can', 'will', 'should', 'could', 'would', 'like', 'get', 'make', 
            'use', 'help', 'new', 'my', 'your', 'this', 'that'
        }
    else:
        # Enhanced stop words list for message content
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'should', 'now', 'could', 'would', 'like', 'get',
            'know', 'think', 'see', 'make', 'go', 'come', 'take', 'use', 'find',
            'give', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
            'user', 'assistant', 'system', 'thought', 'code', 'image', 'attachment'  # ChatGPT specific
        }
    
    jobs[job_id]["progress"] = 50
    
    # Extract meaningful terms - more inclusive
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Also extract key phrases (2-3 word combinations)
    phrases = re.findall(r'\b[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\b', combined_text)
    meaningful_phrases = [p for p in phrases if not any(stop in p.split() for stop in stop_words)]
    
    jobs[job_id]["progress"] = 70
    
    # Count frequencies
    word_counts = Counter(meaningful_words)
    phrase_counts = Counter(meaningful_phrases)
    
    # Get top topics (mix of words and phrases)
    top_words = word_counts.most_common(8)
    top_phrases = phrase_counts.most_common(5)
    
    topics = []
    
    # Add significant words
    for word, count in top_words:
        if count >= 2:  # Must appear at least twice
            topics.append({
                "topic": word.title(),
                "count": count
            })
    
    # Add significant phrases
    for phrase, count in top_phrases:
        if count >= 2:
            topics.append({
                "topic": phrase.title(),
                "count": count
            })
    
    # Sort by count and limit to top 15
    topics = sorted(topics, key=lambda x: x['count'], reverse=True)[:15]
    
    jobs[job_id]["progress"] = 100
    
    return {"topics": topics, "mode": f"simple{mode_suffix}"}

def ingest_stream(file_path: Path, job_id: str, jobs: dict):
    """Background task to process uploaded file - COMPREHENSIVE analysis"""
    try:
        jobs[job_id]["progress"] = 10
        
        # Extract everything from the JSON
        messages, model_usage, conversation_count, total_messages, content_types, conversation_titles = extract_messages_and_models(file_path)
        
        if not messages:
            jobs[job_id]["error"] = "No messages found in file"
            jobs[job_id]["ready"] = True
            return
        
        # Analyze topics using titles when available
        jobs[job_id]["progress"] = 40
        topic_result = simple_topic_analysis(messages, job_id, jobs, conversation_titles)
        
        # Analyze model usage
        jobs[job_id]["progress"] = 80
        model_stats = analyze_model_usage(model_usage)
        
        # Store comprehensive results
        jobs[job_id]["result"] = {
            "topics": topic_result["topics"],
            "topic_mode": topic_result.get("mode", "simple"),
            "models": model_stats,
            "conversation_count": conversation_count,
            "total_messages": total_messages,
            "content_types": dict(content_types),
            "user_messages": len([m for m in messages if m.startswith('[user]')]),
            "assistant_messages": len([m for m in messages if m.startswith('[assistant]')])
        }
        jobs[job_id]["ready"] = True
        jobs[job_id]["message_count"] = len(messages)
        jobs[job_id]["model_requests"] = sum(model_usage.values())
        
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
    
    # Format for charts
    return {
        "series": [topic["count"] for topic in topics],
        "labels": [topic["topic"] for topic in topics],
        "message_count": job.get("message_count", 0),
        "total_messages": result.get("total_messages", 0),
        "content_types": result.get("content_types", {})
    }

def model_stats(job_id: str, jobs: dict):
    """Get comprehensive model usage statistics"""
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    job = jobs[job_id]
    if not job.get("ready"):
        return {"error": "Job not ready"}
    
    if job.get("error"):
        return {"error": job["error"]}
    
    result = job.get("result", {})
    models = result.get("models", [])
    
    return {
        "models": models,
        "total_requests": job.get("model_requests", 0),
        "conversation_count": result.get("conversation_count", 0),
        "total_messages": result.get("total_messages", 0),
        "user_messages": result.get("user_messages", 0),
        "assistant_messages": result.get("assistant_messages", 0),
        "content_types": result.get("content_types", {})
    } 