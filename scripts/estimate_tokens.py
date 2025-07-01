#!/usr/bin/env python3
"""
Analyze actual token usage from conversation data for accurate cost estimation
"""

import json
import statistics
import sys

def estimate_tokens(text):
    """Estimate tokens using GPT tokenization rules (~3.5 chars per token for mixed content)"""
    return len(text) / 3.5

def analyze_conversations(filename, sample_size=100):
    """Analyze conversation token usage"""
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = data.get('conversations', data)[:sample_size]
        token_estimates = []
        
        for conv in conversations:
            title = conv.get('title', '')
            user_content = conv.get('user_content', '')
            
            # Same logic as our embedding code
            if user_content:
                truncated_content = user_content[:2000] if len(user_content) > 2000 else user_content
                combined_text = f"{title}. {truncated_content}"
            else:
                combined_text = title
            
            token_estimate = estimate_tokens(combined_text)
            token_estimates.append(token_estimate)
        
        # Calculate statistics
        min_tokens = min(token_estimates)
        max_tokens = max(token_estimates)
        avg_tokens = statistics.mean(token_estimates)
        median_tokens = statistics.median(token_estimates)
        p75_tokens = sorted(token_estimates)[int(len(token_estimates) * 0.75)]
        p90_tokens = sorted(token_estimates)[int(len(token_estimates) * 0.90)]
        
        print(f"📊 TOKEN ANALYSIS ({len(conversations)} conversations)")
        print(f"   Minimum:  {min_tokens:.0f} tokens")
        print(f"   Average:  {avg_tokens:.0f} tokens")
        print(f"   Median:   {median_tokens:.0f} tokens")
        print(f"   75th %:   {p75_tokens:.0f} tokens")
        print(f"   90th %:   {p90_tokens:.0f} tokens")
        print(f"   Maximum:  {max_tokens:.0f} tokens")
        print()
        
        # Cost calculations
        embedding_price = 0.13 / 1_000_000  # text-embedding-3-large
        
        scenarios = [
            ("Conservative (Average)", avg_tokens),
            ("Realistic (75th %)", p75_tokens),
            ("High Usage (90th %)", p90_tokens)
        ]
        
        print(f"💰 COST ESTIMATES (text-embedding-3-large @ $0.13/1M tokens)")
        for name, tokens_per_conv in scenarios:
            for conv_count in [100, 500, 1000, 5000]:
                total_tokens = tokens_per_conv * conv_count
                cost = total_tokens * embedding_price
                print(f"   {name:20s} - {conv_count:4d} convs: ~{total_tokens:6.0f} tokens = ${cost:.4f}")
        print()
        
        # Recommendations
        print(f"🎯 RECOMMENDATIONS")
        print(f"   • Current estimate (300 tokens): Way too low!")
        print(f"   • Suggested estimate: {p75_tokens:.0f} tokens per conversation")
        print(f"   • For cost warnings: Use 90th percentile ({p90_tokens:.0f} tokens)")
        
        return {
            'avg_tokens': avg_tokens,
            'p75_tokens': p75_tokens,
            'p90_tokens': p90_tokens
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "cleaned_conversations_v2.json"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    analyze_conversations(filename, sample_size) 