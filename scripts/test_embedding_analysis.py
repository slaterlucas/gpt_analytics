#!/usr/bin/env python3
"""
Test script for the new embedding-based topic analysis
"""

import json
import sys
from pathlib import Path

# Add the API module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from app.ingest import embedding_based_topic_analysis

def test_embedding_analysis(cleaned_file: str, num_clusters: int = 10, max_conversations: int = 100, use_llm_naming: bool = True):
    """Test the embedding-based topic analysis"""
    
    print(f"🧪 Testing embedding-based topic analysis")
    print(f"   File: {cleaned_file}")
    print(f"   Clusters: {num_clusters}")
    print(f"   Max conversations: {max_conversations}")
    print(f"   LLM naming: {use_llm_naming}")
    print()
    
    # Load cleaned conversations
    try:
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'conversations' in data:
            conversations = data['conversations']
        elif isinstance(data, list):
            conversations = data
        else:
            print("❌ Invalid file format")
            return
        
        print(f"✅ Loaded {len(conversations)} conversations")
        
        # Run analysis (limiting to max_conversations for testing)
        result = embedding_based_topic_analysis(
            conversations=conversations,
            num_clusters=num_clusters,
            max_conversations=max_conversations,
            use_llm_naming=use_llm_naming
        )
        
        # Display results
        print(f"\n🎯 TOPIC ANALYSIS RESULTS")
        print(f"   Method: {result['method']}")
        print(f"   Model: {result['model_used']}")
        print(f"   Total conversations analyzed: {result['total_conversations']}")
        print(f"   Number of clusters: {result['num_clusters']}")
        print()
        
        print("📊 TOPIC CLUSTERS:")
        for i, cluster_info in enumerate(result['cluster_summary'][:10]):  # Show top 10
            topic_name = cluster_info['topic_name']
            original_name = cluster_info.get('original_topic_name')
            naming_method = cluster_info.get('naming_method', 'keyword_based')
            
            # Show both names if LLM naming was used
            if original_name and naming_method == 'llm_generated':
                print(f"{i+1:2d}. {topic_name:<30} ({cluster_info['conversation_count']:3d} conversations)")
                print(f"    Original: {original_name}")
                print(f"    Keywords: {', '.join(cluster_info['keywords'])}")
            else:
                print(f"{i+1:2d}. {topic_name:<30} ({cluster_info['conversation_count']:3d} conversations)")
                print(f"    Keywords: {', '.join(cluster_info['keywords'])}")
            print()
        
        # Show detailed example for largest cluster
        if result['clusters']:
            largest_cluster_id = result['cluster_summary'][0]['cluster_id']
            largest_cluster = result['clusters'][largest_cluster_id]
            
            print(f"🔍 DETAILED VIEW - LARGEST CLUSTER: {largest_cluster['topic_name']}")
            print(f"   Conversations: {largest_cluster['conversation_count']}")
            print(f"   Keywords: {', '.join(largest_cluster['keywords'])}")
            print(f"   Avg messages per conversation: {largest_cluster['avg_message_count']:.1f}")
            print()
            
            print("   Example conversations:")
            for i, conv in enumerate(largest_cluster['conversations'][:3]):
                title = conv['title'][:60] + "..." if len(conv['title']) > 60 else conv['title']
                print(f"   {i+1}. {title}")
            print()
        
        return result
        
    except FileNotFoundError:
        print(f"❌ File not found: {cleaned_file}")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {cleaned_file}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_embedding_analysis.py <cleaned_conversations.json> [num_clusters] [max_conversations] [use_llm_naming]")
        print("Example: python test_embedding_analysis.py cleaned_conversations_v3.json 15 200 true")
        print("         python test_embedding_analysis.py cleaned_conversations_v3.json 12 100 false")
        sys.exit(1)
    
    cleaned_file = sys.argv[1]
    num_clusters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_conversations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    use_llm_naming = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
    
    # Set OpenAI API key (optional)
    import os
    if 'OPENAI_API_KEY' in os.environ:
        import openai
        openai.api_key = os.environ['OPENAI_API_KEY']
        print(f"✅ Using OpenAI API key from environment")
    else:
        print(f"⚠️  No OpenAI API key found. Will use TF-IDF fallback if embeddings fail.")
    
    test_embedding_analysis(cleaned_file, num_clusters, max_conversations, use_llm_naming)

if __name__ == "__main__":
    main() 