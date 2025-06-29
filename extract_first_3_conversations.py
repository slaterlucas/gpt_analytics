#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def extract_first_3_conversations():
    """Extract the first 3 conversations from conversations.json"""
    
    input_file = Path("/users/lucasslater/desktop/conversations.json")
    output_file = Path("first_3_conversations.json")
    
    try:
        print(f"Reading conversations from: {input_file}")
        
        # Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different possible JSON structures
        conversations = None
        
        if isinstance(data, list):
            # If the root is a list of conversations
            conversations = data[:3]
            print(f"Found {len(data)} total conversations, extracting first 3")
            
        elif isinstance(data, dict):
            # If it's a dictionary, look for common keys that might contain conversations
            possible_keys = ['conversations', 'data', 'messages', 'chats', 'items']
            
            for key in possible_keys:
                if key in data and isinstance(data[key], list):
                    conversations = data[key][:3]
                    print(f"Found {len(data[key])} conversations in '{key}' field, extracting first 3")
                    # Preserve the original structure
                    result = data.copy()
                    result[key] = conversations
                    conversations = result
                    break
            
            if conversations is None:
                print("Available keys in the JSON:")
                for key in data.keys():
                    print(f"  - {key}: {type(data[key])}")
                print("\nCouldn't automatically detect conversation structure.")
                return False
        
        else:
            print(f"Unexpected JSON structure: {type(data)}")
            return False
        
        # Write the first 3 conversations to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully extracted first 3 conversations to: {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = extract_first_3_conversations()
    sys.exit(0 if success else 1) 