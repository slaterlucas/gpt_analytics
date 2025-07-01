import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import openai
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import warnings

# OpenAI Pricing (2024)
OPENAI_PRICING = {
    "text-embedding-3-large": {
        "input": 0.13 / 1_000_000,  # $0.13 per 1M tokens  
        "output": 0.0,  # No output for embeddings
        "description": "Most capable embedding model"
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,   # $0.15 per 1M tokens
        "output": 0.60 / 1_000_000,   # $0.60 per 1M tokens
        "description": "Efficient chat model for topic naming"
    }
}

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
    daily_messages = defaultdict(int)  # Track messages per day
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
                create_time = message.get('create_time')
                
                # Skip system messages that are empty or hidden
                metadata = message.get('metadata', {})
                if metadata.get('is_visually_hidden_from_conversation'):
                    continue
                
                total_messages += 1
                
                # Track daily message counts
                if create_time:
                    try:
                        # Convert Unix timestamp to date string
                        dt = datetime.fromtimestamp(create_time, tz=timezone.utc)
                        date_str = dt.strftime('%Y-%m-%d')
                        daily_messages[date_str] += 1
                    except (ValueError, TypeError):
                        pass  # Skip invalid timestamps
                
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
    
    return messages, model_usage, conversation_count, total_messages, content_types, daily_messages, conversation_titles

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

def simple_topic_analysis(messages, job_id: str, jobs: dict):
    """Dynamic topic discovery - finds actual topics from your conversations"""
    if not messages:
        return {"topics": [], "message": "No messages found", "mode": "simple"}
    
    jobs[job_id]["progress"] = 20
    
    # Clean and prepare text
    all_text = []
    for msg in messages:
        cleaned_msg = re.sub(r'^\[.*?\]\s*', '', msg)  # Remove role prefixes
        cleaned_msg = re.sub(r'\[.*?:\s*', '', cleaned_msg)  # Remove markers
        cleaned_msg = re.sub(r'\]', '', cleaned_msg)
        if len(cleaned_msg.strip()) > 20:  # Only substantial messages
            all_text.append(cleaned_msg)
    
    combined_text = ' '.join(all_text)
    
    jobs[job_id]["progress"] = 40
    
    # Advanced stop words (more comprehensive)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
        'will', 'just', 'should', 'now', 'could', 'would', 'like', 'get', 'know', 'think', 'see',
        'make', 'go', 'come', 'take', 'use', 'find', 'give', 'tell', 'ask', 'work', 'seem', 'feel',
        'try', 'leave', 'call', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these',
        'those', 'user', 'assistant', 'system', 'also', 'need', 'want', 'looking', 'trying',
        'doing', 'getting', 'working', 'thanks', 'please', 'help', 'yes', 'okay', 'sure', 'well',
        'good', 'great', 'nice', 'perfect', 'exactly', 'really', 'actually', 'probably', 'maybe',
        'something', 'anything', 'everything', 'nothing', 'someone', 'anyone', 'everyone'
    }
    
    # Extract meaningful n-grams (1-4 words)
    def extract_ngrams(text, n):
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Get all n-grams
    unigrams = extract_ngrams(combined_text, 1)
    bigrams = extract_ngrams(combined_text, 2) 
    trigrams = extract_ngrams(combined_text, 3)
    fourgrams = extract_ngrams(combined_text, 4)
    
    jobs[job_id]["progress"] = 60
    
    # Filter out stop word combinations and calculate scores
    def score_ngram(ngram, count, total_words):
        words = ngram.split()
        
        # Skip if contains stop words
        if any(word in stop_words for word in words):
            return 0
            
        # Skip if too short or too common
        if len(ngram) < 4 or count < 3:
            return 0
            
        # TF-IDF-like scoring: frequency * inverse document frequency approximation
        # Longer phrases get bonus, rarer phrases get bonus
        length_bonus = len(words) * 1.5  # Prefer multi-word topics
        frequency_score = count / total_words
        rarity_bonus = 1 / (count + 1)  # Prefer not-too-common terms
        
        return frequency_score * length_bonus * (1 + rarity_bonus)
    
    # Count and score all n-grams
    all_ngrams = unigrams + bigrams + trigrams + fourgrams
    ngram_counts = Counter(all_ngrams)
    total_words = len(unigrams)
    
    scored_topics = []
    for ngram, count in ngram_counts.items():
        score = score_ngram(ngram, count, total_words)
        if score > 0:
            scored_topics.append({
                "topic": ngram.title(),
                "count": count,
                "score": score
            })
    
    jobs[job_id]["progress"] = 80
    
    # Sort by score and take top topics
    scored_topics.sort(key=lambda x: x['score'], reverse=True)
    
    # Remove very similar topics (simple deduplication)
    final_topics = []
    used_words = set()
    
    for topic_data in scored_topics:
        topic = topic_data["topic"]
        words = set(topic.lower().split())
        
        # Skip if too much overlap with existing topics
        overlap = len(words.intersection(used_words))
        if overlap < len(words) * 0.7:  # Less than 70% overlap
            final_topics.append({
                "topic": topic,
                "count": topic_data["count"]
            })
            used_words.update(words)
            
        if len(final_topics) >= 15:  # Limit to top 15
            break
    
    # If we don't have enough good topics, add some high-frequency single words
    if len(final_topics) < 8:
        single_word_counts = Counter([word for word in unigrams if word not in stop_words and len(word) > 4])
        for word, count in single_word_counts.most_common(5):
            if count >= 10 and word.title() not in [t["topic"] for t in final_topics]:
                final_topics.append({
                    "topic": word.title(),
                    "count": count
                })
    
    jobs[job_id]["progress"] = 100
    
    return {"topics": final_topics[:12], "mode": "simple"}

def bertopic_analysis(messages, job_id: str, jobs: dict, conversation_titles=None):
    """Advanced semantic topic discovery using BERTopic - TITLE-BASED MODE for efficiency"""
    
    # NEW: Use conversation titles instead of full messages for topic analysis
    if conversation_titles and len(conversation_titles) > 0:
        # Clean and preprocess titles for better topic extraction
        cleaned_titles = []
        for title in conversation_titles:
            # Remove common generic patterns from titles
            cleaned = title.lower()
            
            # Remove generic request patterns
            cleaned = re.sub(r'\b(help with|need help|question about|how to|looking for)\b', '', cleaned)
            cleaned = re.sub(r'\b(request for|asking for|need|want|trying to)\b', '', cleaned)
            cleaned = re.sub(r'\b(interview|internship|application|job|career)\s+(help|advice|question|request)\b', 'career', cleaned)
            
            # Clean up whitespace and keep meaningful content
            cleaned = ' '.join(cleaned.split())
            
            # Only keep titles with meaningful content after cleaning
            if len(cleaned) > 5 and cleaned != title.lower():
                cleaned_titles.append(cleaned)
            else:
                # Keep original if cleaning removed too much
                cleaned_titles.append(title)
        
        documents = cleaned_titles
        print(f"DEBUG: Using {len(documents)} conversation titles for BERTopic analysis")
        mode_suffix = " (Title-Based)"
    else:
        # Fallback to messages if no titles available
        if not messages:
            return {"topics": [], "message": "No messages or titles found", "mode": "bertopic"}
        
        # Clean and prepare documents with better preprocessing
        documents = []
        for msg in messages:
            cleaned_msg = re.sub(r'^\[.*?\]\s*', '', msg)  # Remove role prefixes
            cleaned_msg = re.sub(r'\[.*?:\s*', '', cleaned_msg)  # Remove markers like [Thought:, [Code:
            cleaned_msg = re.sub(r'\]', '', cleaned_msg)  # Remove remaining brackets
            
            # Additional cleaning for better topic extraction
            cleaned_msg = re.sub(r'https?://\S+', '', cleaned_msg)  # Remove URLs
            cleaned_msg = re.sub(r'\b\d+\b', '', cleaned_msg)  # Remove standalone numbers
            cleaned_msg = re.sub(r'[^\w\s]', ' ', cleaned_msg)  # Replace punctuation with spaces
            cleaned_msg = ' '.join(cleaned_msg.split())  # Normalize whitespace
            
            # Additional filtering to remove structural artifacts
            cleaned_msg = re.sub(r'\b(thought|thoughts|seconds|content_type|reasoning_recap|summary|null|memory|result|results)\b', '', cleaned_msg, flags=re.IGNORECASE)
            cleaned_msg = re.sub(r'\b(attachment|image|code|language|multimodal_text|parts|metadata|author|role|timestamp)\b', '', cleaned_msg, flags=re.IGNORECASE)
            cleaned_msg = ' '.join(cleaned_msg.split())  # Normalize whitespace again
            
            if len(cleaned_msg.strip()) > 30:  # Only substantial messages
                documents.append(cleaned_msg.strip())
        
        print(f"DEBUG: Using {len(documents)} message documents for BERTopic analysis")
        mode_suffix = " (Message-Based)"
    
    if len(documents) < 5:
        print(f"DEBUG: Not enough documents for BERTopic ({len(documents)}), falling back to simple")
        return simple_topic_analysis(messages, job_id, jobs)
    
    try:
        jobs[job_id]["progress"] = 20
        print(f"DEBUG: Starting BERTopic analysis for {len(documents)} documents")
        
        jobs[job_id]["progress"] = 40
        
        # For titles, use simpler stop words since titles are already concise
        if conversation_titles:
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'from', 'up', 'about', 'how', 'what', 'why', 'when', 'where', 'can', 'will', 'should',
                'could', 'would', 'like', 'get', 'make', 'use', 'help', 'new', 'my', 'your', 'this', 'that',
                'request', 'question', 'need', 'want', 'looking', 'trying', 'doing', 'getting', 'working', 'inquiry'
            }
        else:
            # Enhanced stop words for message content (existing comprehensive list)
            stop_words = {
                # Basic stop words
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
                'will', 'just', 'should', 'now', 'could', 'would', 'like', 'get', 'know', 'think', 'see',
                'make', 'go', 'come', 'take', 'use', 'find', 'give', 'tell', 'ask', 'work', 'seem', 'feel',
                'try', 'leave', 'call', 'way', 'may', 'say', 'come', 'its', 'our', 'out', 'day', 'has', 'had',
                # Pronouns and common words
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
                # ChatGPT specific
                'user', 'assistant', 'system', 'chatgpt', 'openai', 'gpt', 'ai', 'model',
                # ChatGPT Export Structure Terms (THE MAIN ISSUE)
                'thought', 'thoughts', 'seconds', 'content_type', 'reasoning_recap', 'summary',
                'null', 'memory', 'result', 'results', 'response', 'responses', 'message', 'messages',
                'conversation', 'conversations', 'chat', 'chats', 'export', 'data', 'json',
                'attachment'
                'multimodal_text', 'text', 'parts', 'part', 'content', 'metadata', 'author', 'role',
                'create_time', 'timestamp', 'node', 'mapping', 'id', 'uuid', 'slug', 'model_slug',
                # Common conversation words
                'also', 'need', 'want', 'looking', 'trying', 'doing', 'getting', 'working', 'thanks', 
                'please', 'help', 'yes', 'okay', 'sure', 'well', 'good', 'great', 'nice', 'perfect', 
                'exactly', 'really', 'actually', 'probably', 'maybe', 'something', 'anything', 
                'everything', 'nothing', 'someone', 'anyone', 'everyone', 'thing', 'things',
                # Question words and generic terms
                'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom',
                'are', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'new', 'old', 'first', 'last', 'next', 'previous', 'current', 'recent', 'latest',
                'information', 'question', 'answer', 'example', 'case', 'section', 'item', 'element', 
                'feature', 'option', 'choice', 'different', 'similar', 'same', 'various', 'multiple', 
                'single', 'specific', 'general', 'important', 'useful', 'helpful', 'necessary', 
                'possible', 'available', 'common', 'basic'
            }
        
        # Initialize BERTopic with MUCH more aggressive settings for granular topics
        vectorizer_model = CountVectorizer(
            stop_words=list(stop_words),
            min_df=1,  # Allow words that appear only once
            max_df=0.7,  # More restrictive on common words
            ngram_range=(1, 2),  # Only 1-2 word phrases for titles
            lowercase=True,
            max_features=1000  # Limit vocabulary size
        )
        
        # Use a lightweight sentence transformer for speed
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure BERTopic for conversational data - FIXED: More granular topics
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=3 if conversation_titles else 5,  # MUCH smaller minimum - only 3 titles per topic!
            nr_topics=None,  # Let BERTopic find optimal number of topics
            calculate_probabilities=False,  # Faster without probabilities
            verbose=False
        )
        
        jobs[job_id]["progress"] = 60
        print(f"DEBUG: Fitting BERTopic model...")
        
        # Fit the model and get topics
        topics, probabilities = topic_model.fit_transform(documents)
        
        jobs[job_id]["progress"] = 80
        print(f"DEBUG: BERTopic found {len(set(topics))} topics")
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        # Format results for our API with better topic naming
        bertopic_topics = []
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:  # Skip outlier topic
                continue
                
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # Filter out stop words from topic words and create meaningful names
                filtered_words = []
                for word, score in topic_words[:10]:  # Look at top 10 words
                    if word.lower() not in stop_words and len(word) > 2 and score > 0.01:
                        filtered_words.append((word, score))
                
                if len(filtered_words) >= 1:  # More lenient for titles
                    # Create topic name from top meaningful words
                    if len(filtered_words) >= 3:
                        # Use top 3 words for richer topic names
                        top_words = [word for word, _ in filtered_words[:3]]
                        topic_name = ' + '.join(top_words).title()
                    elif len(filtered_words) >= 2:
                        # Use top 2 words
                        top_words = [word for word, _ in filtered_words[:2]]
                        topic_name = ' + '.join(top_words).title()
                    else:
                        # Use single word
                        topic_name = filtered_words[0][0].title()
                    
                    bertopic_topics.append({
                        "topic": topic_name,
                        "count": int(row['Count']),
                        "words": [word for word, _ in filtered_words[:5]]  # Top 5 meaningful words
                    })
                    
                    print(f"DEBUG: Topic {topic_id}: {topic_name} (count: {row['Count']})")
        
        # Sort by count and limit
        bertopic_topics = sorted(bertopic_topics, key=lambda x: x['count'], reverse=True)[:12]
        
        # HYBRID APPROACH: If BERTopic gives us too few topics, supplement with keyword analysis
        if len(bertopic_topics) < 8:
            print(f"DEBUG: BERTopic returned only {len(bertopic_topics)} topics, supplementing with keyword analysis")
            
            # Get keyword-based topics from titles
            if conversation_titles:
                combined_text = ' '.join(conversation_titles).lower()
                
                # Extract meaningful terms
                words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
                meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
                
                # Get phrases too
                phrases = re.findall(r'\b[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\b', combined_text)
                meaningful_phrases = [p for p in phrases if not any(stop in p.split() for stop in stop_words)]
                
                # Count frequencies
                word_counts = Counter(meaningful_words)
                phrase_counts = Counter(meaningful_phrases)
                
                # Get existing topic names to avoid duplicates
                existing_topics = {topic['topic'].lower() for topic in bertopic_topics}
                
                # Add top keywords that aren't already covered
                for word, count in word_counts.most_common(15):
                    if count >= 3 and word.title() not in existing_topics:
                        bertopic_topics.append({
                            "topic": word.title(),
                            "count": count,
                            "words": [word]
                        })
                        existing_topics.add(word.title().lower())
                        if len(bertopic_topics) >= 15:
                            break
                
                # Add top phrases that aren't already covered
                for phrase, count in phrase_counts.most_common(10):
                    if count >= 2 and phrase.title() not in existing_topics:
                        bertopic_topics.append({
                            "topic": phrase.title(),
                            "count": count,
                            "words": phrase.split()
                        })
                        existing_topics.add(phrase.title().lower())
                        if len(bertopic_topics) >= 15:
                            break
                
                # Re-sort after adding keyword topics
                bertopic_topics = sorted(bertopic_topics, key=lambda x: x['count'], reverse=True)[:15]
                print(f"DEBUG: After hybrid approach, returning {len(bertopic_topics)} topics")
        
        jobs[job_id]["progress"] = 100
        print(f"DEBUG: BERTopic analysis complete, returning {len(bertopic_topics)} topics")
        
        return {"topics": bertopic_topics, "mode": f"bertopic{mode_suffix}"}
        
    except Exception as e:
        print(f"DEBUG: BERTopic analysis failed: {e}")
        print(f"DEBUG: Falling back to simple analysis")
        # Fallback to simple analysis if BERTopic fails
        return simple_topic_analysis(messages, job_id, jobs)

def openai_topic_analysis(messages, job_id: str, jobs: dict, api_key: str):
    """OpenAI-enhanced topic analysis - ADVANCED MODE (TESTING - LIMITED SCOPE)"""
    if not messages:
        return {"topics": [], "message": "No messages found", "mode": "openai"}
    
    try:
        # Configure OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        jobs[job_id]["progress"] = 20
        
        # TESTING LIMITS: Use much smaller samples to save costs
        sample_size = min(10, len(messages))  # Only use first 10 messages for testing
        sample_messages = messages[:sample_size]
        
        # Prepare text for OpenAI analysis - very limited for testing
        sample_text = '\n'.join(sample_messages[:5])  # Only first 5 messages
        
        jobs[job_id]["progress"] = 40
        
        # Create OpenAI prompt for topic analysis - shorter for testing
        prompt = f"""
        Analyze these ChatGPT messages and identify 3-5 main topics. 
        Return ONLY JSON array format: [{{"topic": "Topic Name", "count": 50}}]
        
        Messages (limited sample):
        {sample_text[:500]}
        """
        
        jobs[job_id]["progress"] = 60
        
        # Call OpenAI API with conservative settings for testing
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analyze topics. Return only JSON array."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Reduced from 1000 to save costs
            temperature=0.1   # Lower temperature for more consistent results
        )
        
        jobs[job_id]["progress"] = 80
        
        # Parse OpenAI response
        try:
            topics_text = response.choices[0].message.content.strip()
            print(f"OpenAI response (testing): {topics_text}")  # Debug logging
            
            # Extract JSON from response if it's wrapped in text
            if '[' in topics_text and ']' in topics_text:
                start_idx = topics_text.find('[')
                end_idx = topics_text.rfind(']') + 1
                topics_json = topics_text[start_idx:end_idx]
                topics = json.loads(topics_json)
            else:
                print("No JSON found in OpenAI response, falling back to simple analysis")
                # Fallback to simple analysis
                return simple_topic_analysis(messages, job_id, jobs)
            
            # Validate and clean topics
            validated_topics = []
            for topic in topics:
                if isinstance(topic, dict) and 'topic' in topic and 'count' in topic:
                    validated_topics.append({
                        "topic": str(topic['topic']).title(),
                        "count": int(topic['count'])
                    })
            
            # Sort by count and limit to max 10 for testing
            validated_topics = sorted(validated_topics, key=lambda x: x['count'], reverse=True)[:10]
            
            jobs[job_id]["progress"] = 100
            
            print(f"OpenAI analysis successful: {len(validated_topics)} topics found")
            return {"topics": validated_topics, "mode": "openai"}
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"OpenAI response parsing error: {e}")
            # Fallback to simple analysis
            return simple_topic_analysis(messages, job_id, jobs)
            
    except Exception as e:
        print(f"OpenAI API error: {e}")
        # Fallback to simple analysis
        return simple_topic_analysis(messages, job_id, jobs)

def embedding_based_topic_analysis(conversations: List[Dict], num_clusters: int = 15, max_conversations: int = None, use_llm_naming: bool = True, api_key: str = None) -> Dict[str, Any]:
    """
    Cost-effective topic analysis using OpenAI embeddings + clustering
    
    Args:
        conversations: List of cleaned conversations
        num_clusters: Number of topic clusters to create
        max_conversations: Limit conversations for cost control (None = all)
        use_llm_naming: Whether to use LLM-based topic naming
        api_key: OpenAI API key (optional, uses environment if not provided)
    
    Returns:
        Dictionary with clusters, topics, and metadata
    """
    print(f"🔍 Starting embedding-based topic analysis...")
    
    # Limit conversations for cost control if specified
    if max_conversations and len(conversations) > max_conversations:
        conversations = conversations[:max_conversations]
        print(f"   Limited to {max_conversations} conversations for cost control")
    
    print(f"   Processing {len(conversations)} conversations")
    
    # Prepare text data for embedding
    texts_for_embedding = []
    conversation_metadata = []
    
    for i, conv in enumerate(conversations):
        # Combine title + user content for better topic representation
        title = conv.get('title', '') or f'Conversation {i+1}'
        user_content = conv.get('user_content', '')
        
        # Create a representative text (title gets more weight)
        if user_content:
            # Truncate user content to avoid hitting token limits
            truncated_content = user_content[:2000] if len(user_content) > 2000 else user_content
            combined_text = f"{title}. {truncated_content}"
        else:
            combined_text = title
        
        texts_for_embedding.append(combined_text)
        conversation_metadata.append({
            'index': i,
            'title': title,
            'created_at': conv.get('created_at'),
            'total_chars': conv.get('total_chars', 0),
            'message_count': conv.get('message_count', 0)
        })
    
    # Generate embeddings using OpenAI
    print(f"   Generating embeddings for {len(texts_for_embedding)} texts...")
    try:
        # Setup OpenAI client
        client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        
        # Use text-embedding-3-large for cost efficiency (cheaper than 3-large)
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits
        
        for i in range(0, len(texts_for_embedding), batch_size):
            batch = texts_for_embedding[i:i + batch_size]
            print(f"   Processing embedding batch {i//batch_size + 1}/{(len(texts_for_embedding) + batch_size - 1)//batch_size}")
            
            response = client.embeddings.create(
                model="text-embedding-3-large",  # Cost-effective option
                input=batch,
                encoding_format="float"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
    
    except Exception as e:
        print(f"   ❌ OpenAI embedding failed: {e}")
        print(f"   Falling back to TF-IDF for topic extraction")
        return fallback_tfidf_analysis(conversations, num_clusters)
    
    # Convert to numpy array for clustering
    embeddings_array = np.array(embeddings)
    print(f"   Created embeddings matrix: {embeddings_array.shape}")
    
    # Add larger random noise to prevent identical embeddings (fixes divide by zero)
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 1e-6, embeddings_array.shape)  # Increased noise
    embeddings_array = embeddings_array + noise
    
    # Normalize embeddings to prevent overflow
    embeddings_array = normalize(embeddings_array, norm='l2', axis=1)
    
    # Additional stability check
    embeddings_array = np.clip(embeddings_array, -1.0, 1.0)  # Clip extreme values
    
    # Perform K-Means clustering with warnings suppressed
    print(f"   Clustering into {num_clusters} topics...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=300, tol=1e-6)
        cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Group conversations by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            **conversation_metadata[i],
            'text': texts_for_embedding[i]
        })
    
    # Extract representative keywords for each cluster using TF-IDF
    print(f"   Extracting keywords for each cluster...")
    cluster_topics = {}
    
    for cluster_id, cluster_conversations in clusters.items():
        # Combine all text from this cluster
        cluster_texts = [conv['text'] for conv in cluster_conversations]
        cluster_combined = ' '.join(cluster_texts)
        
        # Extract keywords using TF-IDF
        keywords = extract_cluster_keywords(cluster_texts, top_k=8)
        
        # Generate a topic name from top keywords
        topic_name = generate_topic_name(keywords[:3])
        
        cluster_topics[cluster_id] = {
            'topic_name': topic_name,
            'keywords': keywords,
            'conversation_count': len(cluster_conversations),
            'conversations': cluster_conversations[:5],  # Store top 5 examples
            'total_chars': sum(conv.get('total_chars', 0) for conv in cluster_conversations),
            'avg_message_count': np.mean([conv.get('message_count', 0) for conv in cluster_conversations])
        }
    
    # Sort clusters by size (most conversations first)
    sorted_clusters = sorted(cluster_topics.items(), key=lambda x: x[1]['conversation_count'], reverse=True)
    
    # Create final result
    result = {
        'method': 'embedding_clustering',
        'model_used': 'text-embedding-3-large',
        'num_clusters': num_clusters,
        'total_conversations': len(conversations),
        'clusters': dict(sorted_clusters),
        'cluster_summary': [
            {
                'cluster_id': cluster_id,
                'topic_name': info['topic_name'],
                'conversation_count': info['conversation_count'],
                'keywords': info['keywords'][:5]
            }
            for cluster_id, info in sorted_clusters
        ]
    }
    
    print(f"   ✅ Completed clustering analysis!")
    print(f"   Found {len(clusters)} topic clusters")
    print(f"   Largest cluster: {sorted_clusters[0][1]['conversation_count']} conversations")
    
    # Step 3: LLM-based topic naming (optional)
    if use_llm_naming:
        improved_clusters = generate_llm_topic_names(cluster_topics, api_key)
        # Update the results with improved names
        result['clusters'] = improved_clusters
        result['cluster_summary'] = [
            {
                'cluster_id': cluster_id,
                'topic_name': info['topic_name'],
                'original_topic_name': info.get('original_topic_name'),
                'naming_method': info.get('naming_method', 'llm_generated'),
                'conversation_count': info['conversation_count'],
                'keywords': info['keywords'][:5]
            }
            for cluster_id, info in sorted(improved_clusters.items(), key=lambda x: x[1]['conversation_count'], reverse=True)
        ]
    
    return result

def extract_cluster_keywords(texts: List[str], top_k: int = 8) -> List[str]:
    """Extract representative keywords from a cluster of texts using TF-IDF"""
    if not texts:
        return []
    
    # Adaptive parameters based on cluster size
    cluster_size = len(texts)
    min_df = max(1, min(2, cluster_size // 3))  # Adaptive min_df
    max_df = min(0.8, max(0.5, 1.0 - (1.0 / cluster_size)))  # Adaptive max_df
    
    # Use TF-IDF to find important terms
    vectorizer = TfidfVectorizer(
        max_features=min(500, cluster_size * 50),  # Scale features with cluster size
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=min_df,
        max_df=max_df,
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens, min 2 chars
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        if len(feature_names) == 0:
            # Fallback: simple word frequency if TF-IDF fails
            print(f"   TF-IDF found no features, using word frequency fallback")
            return extract_simple_keywords(texts, top_k)
        
        # Get mean TF-IDF scores across all documents in cluster
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = mean_scores.argsort()[-top_k:][::-1]
        keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
        
        # If still no keywords, use fallback
        if not keywords:
            print(f"   No keywords from TF-IDF scores, using word frequency fallback")
            return extract_simple_keywords(texts, top_k)
        
        return keywords
        
    except Exception as e:
        print(f"   Warning: TF-IDF extraction failed: {e}")
        print(f"   Using simple word frequency fallback")
        return extract_simple_keywords(texts, top_k)

def extract_simple_keywords(texts: List[str], top_k: int = 8) -> List[str]:
    """Fallback keyword extraction using simple word frequency"""
    if not texts:
        return []
    
    from collections import Counter
    import re
    
    # Combine all text and extract words
    combined_text = ' '.join(texts).lower()
    
    # Simple tokenization and filtering
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)  # 3+ letter words only
    
    # Filter out common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 
        'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 
        'see', 'two', 'way', 'who', 'boy', 'did', 'does', 'let', 'put', 'say', 'she', 'too', 'use',
        'this', 'that', 'with', 'have', 'they', 'will', 'been', 'from', 'were', 'said', 'each', 
        'which', 'their', 'time', 'would', 'there', 'could', 'other', 'more', 'very', 'what',
        'just', 'like', 'think', 'know', 'want', 'need', 'good', 'make', 'really', 'much'
    }
    
    # Count words and filter
    word_counts = Counter(word for word in words if word not in stop_words)
    
    # Return top words
    return [word for word, count in word_counts.most_common(top_k)]

def generate_topic_name(keywords: List[str]) -> str:
    """Generate a human-readable topic name from keywords"""
    if not keywords:
        return "General Discussion"
    
    # Simple topic name generation - can be enhanced
    if len(keywords) >= 2:
        return f"{keywords[0].title()} & {keywords[1].title()}"
    else:
        return keywords[0].title()

def fallback_tfidf_analysis(conversations: List[Dict], num_clusters: int) -> Dict[str, Any]:
    """Fallback analysis using only TF-IDF when embeddings fail"""
    print(f"   Using TF-IDF fallback analysis...")
    
    # Extract text content
    texts = []
    for conv in conversations:
        title = conv.get('title', '')
        user_content = conv.get('user_content', '')
        combined = f"{title}. {user_content[:1000]}" if user_content else title
        texts.append(combined)
    
    # TF-IDF clustering (simplified)
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        kmeans = KMeans(n_clusters=min(num_clusters, len(texts)), random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group conversations by cluster and create proper structure
        raw_clusters = {}
        for i, label in enumerate(labels):
            if label not in raw_clusters:
                raw_clusters[label] = []
            raw_clusters[label].append(conversations[i])
        
        # Convert to expected format with topic names and counts
        formatted_clusters = {}
        for cluster_id, cluster_conversations in raw_clusters.items():
            # Extract simple keywords from titles
            titles = [conv.get('title', '') for conv in cluster_conversations]
            combined_titles = ' '.join(titles).lower()
            
            # Simple keyword extraction from titles
            words = combined_titles.split()
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
            keywords = [word for word in words if len(word) > 2 and word not in common_words]
            top_keywords = list(set(keywords))[:3]  # Top 3 unique keywords
            
            # Generate topic name
            if top_keywords:
                topic_name = ' '.join(word.title() for word in top_keywords[:2])
            else:
                topic_name = f"Topic {cluster_id + 1}"
            
            formatted_clusters[cluster_id] = {
                'topic_name': topic_name,
                'keywords': top_keywords,
                'conversation_count': len(cluster_conversations),
                'conversations': cluster_conversations[:5],  # Top 5 examples
                'naming_method': 'tfidf_fallback'
            }
        
        return {
            'method': 'tfidf_fallback',
            'model_used': 'TF-IDF (fallback)',
            'num_clusters': len(formatted_clusters),
            'total_conversations': len(conversations),
            'clusters': formatted_clusters
        }
    
    except Exception as e:
        print(f"   ❌ Fallback analysis also failed: {e}")
        # Return a simple structure that won't break the processing
        return {
            'method': 'failed',
            'error': str(e),
            'clusters': {},
            'total_conversations': len(conversations)
        }

def generate_llm_topic_names(cluster_topics: Dict, api_key: str = None) -> Dict:
    """
    Step 3: Use GPT-4o mini to generate meaningful topic names from clusters
    
    Args:
        cluster_topics: Dictionary of cluster analysis results
        api_key: OpenAI API key (optional, uses environment if not provided)
    
    Returns:
        Updated cluster_topics with improved topic names
    """
    print(f"🎯 Step 3: Generating LLM-based topic names...")
    
    if not api_key:
        import os
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("   ⚠️  No API key available, keeping keyword-based names")
        return cluster_topics
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        updated_clusters = {}
        
        for cluster_id, cluster_info in cluster_topics.items():
            print(f"   Naming cluster {cluster_id}...")
            
            # Prepare data for the LLM
            keywords = cluster_info['keywords'][:8]  # Top 8 keywords
            representative_conversations = cluster_info['conversations'][:5]  # Top 5 conversations
            
            # Create a concise prompt
            prompt = create_topic_naming_prompt(keywords, representative_conversations, cluster_info['conversation_count'])
            
            try:
                # Use GPT-4o mini for cost-effective naming
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing conversation topics and creating concise, descriptive topic names. Generate a clear, specific topic name (2-4 words) that captures the essence of the conversations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.3  # Lower temperature for consistent naming
                )
                
                # Extract the topic name and emoji
                response_text = response.choices[0].message.content.strip()
                
                # Parse the response format: "Topic: [name]\nEmoji: [emoji]"
                lines = response_text.split('\n')
                llm_topic_name = cluster_info['topic_name']  # fallback
                llm_emoji = '💡'  # fallback
                
                for line in lines:
                    if line.startswith('Topic:'):
                        llm_topic_name = line.replace('Topic:', '').strip()
                        llm_topic_name = clean_topic_name(llm_topic_name)
                    elif line.startswith('Emoji:'):
                        llm_emoji = line.replace('Emoji:', '').strip()
                
                # Update the cluster info
                updated_cluster = cluster_info.copy()
                updated_cluster['original_topic_name'] = cluster_info['topic_name']  # Keep original
                updated_cluster['topic_name'] = llm_topic_name
                updated_cluster['emoji'] = llm_emoji
                updated_cluster['naming_method'] = 'llm_generated'
                
                updated_clusters[cluster_id] = updated_cluster
                
                print(f"   {cluster_info['topic_name']} → {llm_emoji} {llm_topic_name}")
                
            except Exception as e:
                print(f"   ❌ Failed to name cluster {cluster_id}: {e}")
                # Keep original name if LLM naming fails
                cluster_info['naming_method'] = 'keyword_fallback'
                updated_clusters[cluster_id] = cluster_info
        
        print(f"   ✅ Generated {len(updated_clusters)} topic names")
        return updated_clusters
        
    except Exception as e:
        print(f"   ❌ LLM topic naming failed: {e}")
        print("   Keeping original keyword-based names")
        return cluster_topics

def create_topic_naming_prompt(keywords: List[str], conversations: List[Dict], conversation_count: int) -> str:
    """Create an effective prompt for topic naming with emoji"""
    
    # Format keywords
    keywords_text = ", ".join(keywords)
    
    # Format representative conversations (titles + snippets)
    conversation_examples = []
    for i, conv in enumerate(conversations[:3]):  # Use top 3 for conciseness
        title = conv.get('title', f'Conversation {i+1}')
        text_snippet = conv.get('text', '')[:200]  # First 200 chars
        conversation_examples.append(f"• {title}: {text_snippet}...")
    
    conversations_text = "\n".join(conversation_examples)
    
    prompt = f"""Based on the following information, generate a topic name (2-4 words) and a single relevant emoji:

KEYWORDS: {keywords_text}

CONVERSATION COUNT: {conversation_count} conversations

REPRESENTATIVE CONVERSATIONS:
{conversations_text}

Generate a specific, descriptive topic name and choose the most fitting emoji. Examples:
- "Python Programming Help" → "💻"
- "Travel Planning" → "✈️"  
- "Career Advice" → "💼"
- "Math Problems" → "🔢"
- "Game Development" → "🎮"

Respond in exactly this format:
Topic: [topic name]
Emoji: [single emoji]"""

    return prompt

def clean_topic_name(raw_name: str) -> str:
    """Clean up the LLM-generated topic name"""
    # Remove common prefixes/suffixes
    cleaned = raw_name.strip()
    
    # Remove quotes
    cleaned = cleaned.strip('"\'""''')
    
    # Remove common phrases
    prefixes_to_remove = [
        "topic:", "topic name:", "the topic is:", "this is about:",
        "conversations about:", "discussions on:", "talks about:"
    ]
    
    cleaned_lower = cleaned.lower()
    for prefix in prefixes_to_remove:
        if cleaned_lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    # Capitalize properly
    cleaned = cleaned.title()
    
    # Limit length
    if len(cleaned) > 50:
        cleaned = cleaned[:47] + "..."
    
    return cleaned or "General Discussion"

def estimate_analysis_cost(num_conversations: int, num_clusters: int = 25, use_llm_naming: bool = True) -> Dict[str, Any]:
    """
    Estimate the cost of running embedding-based topic analysis
    
    Args:
        num_conversations: Number of conversations to analyze
        num_clusters: Number of topic clusters to generate
        use_llm_naming: Whether to use LLM for topic naming
    
    Returns:
        Dict with cost breakdown and estimates
    """
    
    # Embedding cost estimation using text-embedding-3-large
    # Based on analysis of 500 real conversations:
    # - Median: 71 tokens, Average: 181 tokens
    # - 75th percentile: 265 tokens (realistic estimate)
    # - 90th percentile: 579 tokens (high-usage scenarios)
    realistic_tokens_per_conversation = 265   # 75th percentile from real data
    high_usage_tokens_per_conversation = 579  # 90th percentile from real data
    
    # Use realistic estimate for baseline cost
    total_embedding_tokens = num_conversations * realistic_tokens_per_conversation
    embedding_cost = total_embedding_tokens * OPENAI_PRICING["text-embedding-3-large"]["input"]
    
    # Calculate high-usage scenario for warnings
    total_high_usage_tokens = num_conversations * high_usage_tokens_per_conversation
    high_usage_embedding_cost = total_high_usage_tokens * OPENAI_PRICING["text-embedding-3-large"]["input"]
    
    # LLM naming cost estimation (if enabled)
    llm_cost = 0.0
    if use_llm_naming:
        # Each cluster gets ~100 input tokens (keywords + examples) + ~20 output tokens (topic name)
        input_tokens_per_cluster = 100
        output_tokens_per_cluster = 20
        
        total_llm_input_tokens = num_clusters * input_tokens_per_cluster
        total_llm_output_tokens = num_clusters * output_tokens_per_cluster
        
        llm_cost = (
            total_llm_input_tokens * OPENAI_PRICING["gpt-4o-mini"]["input"] +
            total_llm_output_tokens * OPENAI_PRICING["gpt-4o-mini"]["output"]
        )
    
    total_cost = embedding_cost + llm_cost
    high_usage_total_cost = high_usage_embedding_cost + llm_cost
    
    # Create cost breakdown
    cost_breakdown = {
        "total_conversations": num_conversations,
        "num_clusters": num_clusters,
        "use_llm_naming": use_llm_naming,
        "embedding_model": "text-embedding-3-large",
        "costs": {
            "embeddings": {
                "model": "text-embedding-3-large",
                "tokens": total_embedding_tokens,
                "cost": embedding_cost,
                "description": f"text-embedding-3-large for {num_conversations} conversations",
                "tokens_per_conversation": realistic_tokens_per_conversation,
                "estimation_note": "Based on 75th percentile of real conversation data"
            },
            "llm_naming": {
                "input_tokens": total_llm_input_tokens if use_llm_naming else 0,
                "output_tokens": total_llm_output_tokens if use_llm_naming else 0,
                "cost": llm_cost,
                "description": f"gpt-4o-mini for naming {num_clusters} clusters" if use_llm_naming else "Disabled"
            },
            "total": {
                "cost": total_cost,
                "formatted": f"${total_cost:.4f}" if total_cost < 0.01 else f"${total_cost:.3f}"
            },
            "high_usage_scenario": {
                "embedding_tokens": total_high_usage_tokens,
                "embedding_cost": high_usage_embedding_cost,
                "total_cost": high_usage_total_cost,
                "cost_difference": high_usage_total_cost - total_cost,
                "description": "90th percentile usage (content-rich conversations)",
                "tokens_per_conversation": high_usage_tokens_per_conversation
            }
        },
        "cost_per_conversation": total_cost / num_conversations if num_conversations > 0 else 0,
        "warnings": []
    }
    
    # Add warnings based on cost and scale
    if total_cost > 5.0:
        cost_breakdown["warnings"].append("High cost analysis - consider reducing conversation count")
    elif total_cost > 1.0:
        cost_breakdown["warnings"].append("Moderate cost - review settings before proceeding")
    
    if num_conversations > 5000:
        cost_breakdown["warnings"].append("Large dataset - processing may take several minutes")
    
    # Add cost range warning for high-usage scenarios
    cost_difference = high_usage_total_cost - total_cost
    if cost_difference > 0.02:  # More than 2 cents difference
        cost_breakdown["warnings"].append(
            f"Actual costs may be higher ({cost_breakdown['costs']['total']['formatted']} - "
            f"${high_usage_total_cost:.3f}) depending on conversation content length"
        )
    
    return cost_breakdown

def ingest_stream(file_path: Path, job_id: str, jobs: dict, api_key: Optional[str] = None):
    """Background task to process uploaded file - COMPREHENSIVE analysis"""
    try:
        print(f"DEBUG: Starting ingest_stream for job {job_id}")
        jobs[job_id]["progress"] = 10
        print(f"DEBUG: Set progress to 10 for job {job_id}")
        
        # Extract everything from the JSON
        print(f"DEBUG: Extracting messages from {file_path}")
        messages, model_usage, conversation_count, total_messages, content_types, daily_messages, conversation_titles = extract_messages_and_models(file_path)
        print(f"DEBUG: Extracted {len(messages)} messages, {conversation_count} conversations, {len(conversation_titles)} titles")
        
        if not messages:
            print(f"DEBUG: No messages found for job {job_id}")
            jobs[job_id]["error"] = "No messages found in file"
            jobs[job_id]["ready"] = True
            return
        
        # Analyze topics based on whether API key is provided
        jobs[job_id]["progress"] = 40
        print(f"DEBUG: Set progress to 40 for job {job_id}, analyzing topics...")
        
        if api_key and api_key.strip():
            print(f"DEBUG: Using embedding-based analysis for job {job_id}")
            # Convert raw messages to conversation format for embedding analysis
            conversations = []
            for i, title in enumerate(conversation_titles):
                # Create a simplified conversation object
                user_content = ' '.join([msg for msg in messages if msg.startswith('[user]')])[:2000]  # Limit content
                conversations.append({
                    'title': title,
                    'user_content': user_content,
                    'total_chars': len(user_content),
                    'message_count': len([msg for msg in messages if msg.startswith('[user]') or msg.startswith('[assistant]')]) // len(conversation_titles) if conversation_titles else 1
                })
            
            # Use embedding-based analysis with your full dataset
            embedding_result = embedding_based_topic_analysis(
                conversations=conversations,
                num_clusters=min(25, len(conversations)),  # Increased for more granular topics
                use_llm_naming=True,
                api_key=api_key.strip()
            )
            
            # Convert embedding result to expected format
            topics = []
            clusters = embedding_result.get('clusters', {})
            
            if isinstance(clusters, dict):
                for cluster_id, cluster_info in clusters.items():
                    if isinstance(cluster_info, dict) and 'topic_name' in cluster_info and 'conversation_count' in cluster_info:
                        topics.append({
                            'topic': cluster_info['topic_name'],
                            'count': cluster_info['conversation_count']
                        })
                    else:
                        print(f"DEBUG: Invalid cluster_info structure for cluster {cluster_id}: {type(cluster_info)}")
            else:
                print(f"DEBUG: Invalid clusters structure: {type(clusters)}")
                # Fallback to empty topics if structure is wrong
                topics = []
            
            topic_result = {
                'topics': sorted(topics, key=lambda x: x['count'], reverse=True),
                'mode': 'embedding_clustering'
            }
        else:
            print(f"DEBUG: Using BERTopic analysis for job {job_id}")
            # Use BERTopic semantic analysis with conversation titles
            topic_result = bertopic_analysis(messages, job_id, jobs, conversation_titles)
        
        print(f"DEBUG: Topic analysis complete for job {job_id}, found {len(topic_result.get('topics', []))} topics")
        
        # Analyze model usage
        jobs[job_id]["progress"] = 80
        print(f"DEBUG: Set progress to 80 for job {job_id}, analyzing models...")
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
            "assistant_messages": len([m for m in messages if m.startswith('[assistant]')]),
            "daily_messages": dict(daily_messages)  # Add daily message data
        }
        jobs[job_id]["ready"] = True
        jobs[job_id]["message_count"] = len(messages)
        jobs[job_id]["model_requests"] = sum(model_usage.values())
        jobs[job_id]["progress"] = 100
        
        print(f"DEBUG: Processing complete for job {job_id}, marking as ready")
        
        # Clean up file
        if file_path.exists():
            file_path.unlink()
        print(f"DEBUG: Cleaned up file for job {job_id}")
        
    except Exception as e:
        print(f"DEBUG: Error in ingest_stream for job {job_id}: {e}")
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
        "emojis": [topic.get("emoji", "💡") for topic in topics],  # Include AI-generated emojis
        "message_count": job.get("message_count", 0),
        "total_messages": result.get("total_messages", 0),
        "content_types": result.get("content_types", {}),
        "topic_mode": result.get("topic_mode", "simple")
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

def daily_activity(job_id: str, jobs: dict):
    """Get daily message activity for time series chart"""
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    job = jobs[job_id]
    if not job.get("ready"):
        return {"error": "Job not ready"}
    
    if job.get("error"):
        return {"error": job["error"]}
    
    result = job.get("result", {})
    daily_data = result.get("daily_messages", {})
    
    if not daily_data:
        return {
            "dates": [],
            "counts": [],
            "total_days": 0,
            "avg_per_day": 0,
            "peak_day": None,
            "peak_count": 0
        }
    
    # Sort by date
    sorted_data = sorted(daily_data.items())
    dates = [item[0] for item in sorted_data]
    counts = [item[1] for item in sorted_data]
    
    # Calculate stats
    total_days = len(dates)
    total_messages = sum(counts)
    avg_per_day = round(total_messages / total_days, 1) if total_days > 0 else 0
    
    # Find peak day
    if counts:
        peak_idx = counts.index(max(counts))
        peak_day = dates[peak_idx]
        peak_count = counts[peak_idx]
    else:
        peak_day = None
        peak_count = 0
    
    return {
        "dates": dates,
        "counts": counts,
        "total_days": total_days,
        "avg_per_day": avg_per_day,
        "peak_day": peak_day,
        "peak_count": peak_count,
        "total_messages": total_messages
    } 