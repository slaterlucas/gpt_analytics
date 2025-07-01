#!/usr/bin/env python3
"""
Test the updated cost estimation function
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from app.ingest import estimate_analysis_cost

def test_cost_estimation():
    """Test cost estimation with different conversation counts"""
    
    print("🧪 TESTING UPDATED COST ESTIMATION")
    print("   Using text-embedding-3-large with realistic token counts")
    print("   Baseline: 265 tokens/conv (75th %), High-usage: 579 tokens/conv (90th %)")
    print()
    
    test_cases = [100, 500, 1000, 2000, 5000]
    
    for count in test_cases:
        result = estimate_analysis_cost(count, 15, True)
        
        baseline_cost = result['costs']['total']['formatted']
        high_usage_cost = result['costs']['high_usage_scenario']['total_cost']
        cost_range = f"{baseline_cost} - ${high_usage_cost:.3f}"
        
        embedding_tokens = result['costs']['embeddings']['tokens']
        high_usage_tokens = result['costs']['high_usage_scenario']['embedding_tokens']
        
        print(f"{count:4d} conversations:")
        print(f"     Cost range: {cost_range}")
        print(f"     Embedding tokens: {embedding_tokens:,} - {high_usage_tokens:,}")
        
        if result['warnings']:
            print(f"     Warnings: {'; '.join(result['warnings'])}")
        print()

if __name__ == "__main__":
    test_cost_estimation() 