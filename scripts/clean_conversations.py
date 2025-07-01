#!/usr/bin/env python3
"""
Clean and condense conversations.json for topic analysis.
Removes unnecessary fields and combines all messages per conversation.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

def is_filler_message(text: str) -> bool:
    """Check if a message is likely conversational filler that should be filtered out"""
    if not text:
        return True
    
    # Clean and normalize the text
    clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    # Count meaningful words (alphanumeric)
    words = [word for word in clean_text.split() if word.isalnum()]
    
    # Filter out messages with fewer than 5 words
    if len(words) < 5:
        return True
    
    return False

def extract_text_content(content: Dict[str, Any]) -> str:
    """Extract meaningful text from various content types"""
    if not content:
        return ""
    
    text_parts = []
    content_type = content.get('content_type', 'text')
    
    if content_type == 'text':
        parts = content.get('parts', [])
        if isinstance(parts, list):
            text_parts.extend([str(part).strip() for part in parts if part])
    
    elif content_type == 'thoughts':
        thoughts = content.get('thoughts', [])
        if isinstance(thoughts, list):
            for thought in thoughts:
                if isinstance(thought, dict):
                    summary = thought.get('summary', '').strip()
                    content_text = thought.get('content', '').strip()
                    if summary and summary != content_text:
                        text_parts.append(f"[Summary: {summary}]")
                    if content_text:
                        text_parts.append(content_text)
    
    elif content_type == 'code':
        code_text = content.get('text', '').strip()
        language = content.get('language', 'unknown')
        if code_text:
            # Truncate very long code blocks
            if len(code_text) > 500:
                code_text = code_text[:500] + "..."
            text_parts.append(f"[Code ({language}): {code_text}]")
    
    elif content_type == 'multimodal_text':
        parts = content.get('parts', [])
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    if part.get('content_type') == 'text':
                        text = part.get('text', '').strip()
                        if text:
                            text_parts.append(text)
                    elif part.get('content_type') == 'image_asset_pointer':
                        text_parts.append("[Image]")
                elif part:
                    text_parts.append(str(part).strip())
    
    # Fallback: try to extract any text we can find
    if not text_parts:
        if 'parts' in content:
            parts = content['parts']
            if isinstance(parts, list):
                text_parts.extend([str(part).strip() for part in parts if part])
        elif 'text' in content:
            text_parts.append(str(content['text']).strip())
    
    return ' '.join(text_parts).strip()

def clean_conversation(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single conversation for topic analysis"""
    
    # Extract basic info
    title = conv.get('title', '').strip()
    create_time = conv.get('create_time')
    mapping = conv.get('mapping', {})
    
    # Find the root of the conversation tree
    root_id = None
    for node_id, node in mapping.items():
        if node and node.get('parent') is None:
            root_id = node_id
            break
    
    if not root_id:
        # Fallback: look for 'client-created-root' or similar
        for potential_root in ['client-created-root', 'root']:
            if potential_root in mapping:
                root_id = potential_root
                break
    
    if not root_id:
        print(f"Warning: Could not find root node for conversation: {title}")
        return {}
    
    # Walk the conversation tree in order
    def walk_tree(node_id: str, visited: set, stats: dict = None) -> List[Dict[str, str]]:
        """Walk the conversation tree and collect messages in order"""
        if node_id in visited or node_id not in mapping:
            return []
        
        if stats is None:
            stats = {'total_messages': 0, 'filtered_filler': 0}
        
        visited.add(node_id)
        messages = []
        node = mapping[node_id]
        
        if not node:
            return []
        
        # Process current node's message
        if 'message' in node and node['message']:
            message = node['message']
            author = message.get('author', {})
            role = author.get('role')
            content = message.get('content', {})
            
            # Skip system messages and hidden messages  
            metadata = message.get('metadata', {})
            if (role not in ['user', 'assistant'] or 
                metadata.get('is_visually_hidden_from_conversation') or
                not content):
                pass  # Skip but continue to children
            else:
                # Extract text content
                text = extract_text_content(content)
                if text and len(text.strip()) >= 5:
                    # Clean up the text
                    text = re.sub(r'\s+', ' ', text).strip()
                    stats['total_messages'] += 1
                    
                    # Filter out filler messages
                    if not is_filler_message(text):
                        messages.append({
                            'role': role,
                            'content': text,
                            'timestamp': message.get('create_time')
                        })
                    else:
                        stats['filtered_filler'] += 1
        
        # Process children in order
        children = node.get('children', [])
        for child_id in children:
            messages.extend(walk_tree(child_id, visited, stats))
        
        return messages
    
    # Extract all messages in conversation order
    stats = {'total_messages': 0, 'filtered_filler': 0}
    all_messages = walk_tree(root_id, set(), stats)
    
    # Group messages by role while preserving order information
    user_messages = []
    assistant_messages = []
    
    for msg in all_messages:
        if msg['role'] == 'user':
            user_messages.append(msg['content'])
        elif msg['role'] == 'assistant':
            assistant_messages.append(msg['content'])
    
    # Combine all messages by role into single blocks
    user_content = ' '.join(user_messages).strip()
    assistant_content = ' '.join(assistant_messages).strip()
    
    # Create cleaned conversation
    cleaned = {
        'title': title if title and title != 'Untitled' else None,
        'created_at': datetime.fromtimestamp(create_time, tz=timezone.utc).isoformat() if create_time else None,
        'user_content': user_content if user_content else None,
        'assistant_content': assistant_content if assistant_content else None,
        'total_chars': len(user_content) + len(assistant_content),
        'message_count': len(all_messages)
    }
    
    # Remove None values
    cleaned_result = {k: v for k, v in cleaned.items() if v is not None}
    
    # Return both cleaned conversation and stats
    return cleaned_result, stats

def clean_conversations_file(input_file: Path, output_file: Path):
    """Clean the entire conversations file"""
    
    print(f"Reading from: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Handle different JSON structures
    conversations = []
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict):
        # Look for conversations in common keys
        for key in ['conversations', 'data', 'messages', 'chats']:
            if key in data and isinstance(data[key], list):
                conversations = data[key]
                break
        else:
            # Assume the dict itself is a single conversation
            conversations = [data]
    
    print(f"Found {len(conversations)} conversations to clean")
    
    # Clean each conversation
    cleaned_conversations = []
    skipped = 0
    total_filtered_filler = 0
    total_original_messages = 0
    
    for i, conv in enumerate(conversations):
        try:
            cleaned = clean_conversation(conv)
            
            # Track filtering stats if available
            if isinstance(cleaned, tuple):
                cleaned, conv_stats = cleaned
                total_filtered_filler += conv_stats.get('filtered_filler', 0)
                total_original_messages += conv_stats.get('total_messages', 0)
            
            # Only keep conversations with meaningful content
            if (cleaned.get('title') or 
                (cleaned.get('user_content') and len(cleaned['user_content']) > 20) or
                (cleaned.get('assistant_content') and len(cleaned['assistant_content']) > 20)):
                cleaned_conversations.append(cleaned)
            else:
                skipped += 1
                
        except Exception as e:
            print(f"Error cleaning conversation {i}: {e}")
            skipped += 1
    
    print(f"Cleaned {len(cleaned_conversations)} conversations, skipped {skipped}")
    
    # Sort by creation date if available
    cleaned_conversations.sort(
        key=lambda x: x.get('created_at', ''),
        reverse=True
    )
    
    # Save cleaned data
    output_data = {
        'conversations': cleaned_conversations,
        'metadata': {
            'total_conversations': len(cleaned_conversations),
            'total_chars': sum(conv.get('total_chars', 0) for conv in cleaned_conversations),
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'original_file': str(input_file)
        }
    }
    
    print(f"Writing cleaned data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully cleaned conversations!")
    print(f"   Original: {len(conversations)} conversations")
    print(f"   Cleaned:  {len(cleaned_conversations)} conversations")
    print(f"   Total characters: {output_data['metadata']['total_chars']:,}")
    
    if total_original_messages > 0:
        print(f"   Filtered out {total_filtered_filler:,} filler messages ({total_filtered_filler/total_original_messages*100:.1f}% of all messages)")
        print(f"   Kept {total_original_messages - total_filtered_filler:,} meaningful messages")
    
    return True

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python clean_conversations.py <input_file> [output_file]")
        print("Example: python clean_conversations.py conversations.json cleaned_conversations.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('cleaned_conversations.json')
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    success = clean_conversations_file(input_file, output_file)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 