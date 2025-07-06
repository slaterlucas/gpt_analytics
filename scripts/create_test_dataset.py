#!/usr/bin/env python3
"""
Create a test dataset with 200 conversations for topic analysis testing
"""

import json
import random
import os
from pathlib import Path

def create_test_dataset():
    # Look for conversations.json on desktop
    desktop_path = Path.home() / "Desktop"
    input_file = desktop_path / "conversations.json"
    
    if not input_file.exists():
        print(f"❌ Could not find conversations.json on desktop: {input_file}")
        print("Please make sure conversations.json is on your desktop")
        return
    
    try:
        print(f"📖 Reading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract conversations
        conversations = []
        if isinstance(data, list):
            conversations = data
        elif isinstance(data, dict):
            # Try different possible structures
            if 'conversations' in data:
                conversations = data['conversations']
            elif 'data' in data:
                conversations = data['data']
            else:
                # Assume the values are conversations
                conversations = list(data.values())
        
        print(f"📊 Found {len(conversations)} total conversations")
        
        if len(conversations) < 200:
            print(f"⚠️  Only {len(conversations)} conversations available, using all of them")
            sample_conversations = conversations
        else:
            # Randomly sample 200 conversations
            random.seed(42)  # For reproducible sampling
            sample_conversations = random.sample(conversations, 200)
            print(f"🎯 Sampled 200 random conversations for testing")
        
        # Create output file
        output_file = Path(__file__).parent.parent / "test_conversations.json"
        
        # Preserve original structure but with sampled conversations
        if isinstance(data, list):
            output_data = sample_conversations
        elif isinstance(data, dict):
            output_data = data.copy()
            if 'conversations' in data:
                output_data['conversations'] = sample_conversations
            elif 'data' in data:
                output_data['data'] = sample_conversations
            else:
                # Create a simple structure
                output_data = sample_conversations
        else:
            output_data = sample_conversations
        
        # Save test dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created test dataset: {output_file}")
        print(f"📁 Size: {len(sample_conversations)} conversations")
        
        # Print some stats about the sample
        if sample_conversations and isinstance(sample_conversations[0], dict):
            has_titles = sum(1 for c in sample_conversations if c.get('title'))
            print(f"📝 Conversations with titles: {has_titles}")
            
            # Show a few example titles
            titles_with_content = [c.get('title', 'No title') for c in sample_conversations[:5] if c.get('title')]
            if titles_with_content:
                print(f"📋 Sample titles:")
                for i, title in enumerate(titles_with_content[:3], 1):
                    print(f"   {i}. {title[:60]}{'...' if len(title) > 60 else ''}")
        
        print(f"\n🧪 Ready for testing! Upload {output_file.name} to test topic analysis.")
        
    except json.JSONDecodeError as e:
        print(f"❌ Error reading JSON file: {e}")
    except Exception as e:
        print(f"❌ Error creating test dataset: {e}")

if __name__ == "__main__":
    create_test_dataset() 