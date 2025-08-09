#!/usr/bin/env python3
"""
Working configuration finder that properly uses the calculator API.
"""

import json
import time
from pathlib import Path
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import AirdropParameters

def analyze_configuration(calculator, supply, price, airdrop_pct):
    """Analyze a specific configuration for profitability."""
    try:
        # Use the calculator methods with the specific parameters
        # For user segment analysis, we need to check each segment individually
        
        profitable_segments = 0
        total_segments = len(calculator.user_segments)
        segment_details = []
        
        for segment in calculator.user_segments:
            # Use average capital for the segment
            avg_capital = (segment.min_capital + segment.max_capital) / 2
            
            # Calculate minimum profitable tokens needed
            min_tokens = calculator.calculate_min_profitable_tokens(avg_capital)
            
            # Estimate allocation for this capital amount
            est_allocation = calculator.estimate_user_allocation(avg_capital)
            
            # Check if profitable (allocation >= minimum needed)
            is_profitable = est_allocation >= min_tokens
            roi = ((est_allocation * price) - avg_capital) / avg_capital * 100 if avg_capital > 0 else 0
            
            segment_details.append({
                'name': segment.name,
                'avg_capital': avg_capital,
                'min_tokens': min_tokens,
                'est_allocation': est_allocation,
                'profitable': is_profitable,
                'roi': roi
            })
            
            if is_profitable:
                profitable_segments += 1
        
        # Calculate overall metrics
        profitable_percentage = (profitable_segments / total_segments) * 100
        avg_roi = sum(s['roi'] for s in segment_details) / len(segment_details)
        
        # Check if all users are profitable
        all_profitable = profitable_segments == total_segments
        
        return {
            'all_profitable': all_profitable,
            'profitable_percentage': profitable_percentage,
            'avg_roi': avg_roi,
            'segments': segment_details,
            'market_cap': supply * price
        }
        
    except Exception as e:
        print(f"Error analyzing config: {e}")
        return None

def find_profitable_configs():
    """Find configurations where all users are profitable with given constraints."""
    
    print("🔍 Working Profitable Configuration Finder")
    print("=" * 50)
    print("Constraints:")
    print("- All users profitable")
    print("- Price ≥ $0.50")
    print("- Opportunity cost = 5%")
    print("- Airdrop % ≥ 40%")
    print("- Volatility = 30%")
    print("- Certainty = 100%")
    print()
    
    # Initialize components
    solver = EnhancedZ3Solver()
    calculator = AirdropCalculator(solver)
    
    # Search ranges
    supplies = [100_000_000, 500_000_000, 1_000_000_000, 2_000_000_000]
    prices = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00]
    airdrop_percents = [40, 50, 60, 70, 80, 90]
    
    profitable_configs = []
    total_tested = 0
    
    print("Searching configurations...")
    start_time = time.time()
    
    for supply in supplies:
        for price in prices:
            for airdrop_pct in airdrop_percents:
                total_tested += 1
                
                print(f"Testing: {supply:,} tokens × ${price} @ {airdrop_pct}% airdrop", end=" ... ")
                
                result = analyze_configuration(calculator, supply, price, airdrop_pct)
                
                if result and result['all_profitable']:
                    config = {
                        'supply': supply,
                        'price': price,
                        'airdrop_percent': airdrop_pct,
                        'market_cap': result['market_cap'],
                        'avg_roi': result['avg_roi'],
                        'profitable_percentage': result['profitable_percentage'],
                        'segments': result['segments']
                    }
                    profitable_configs.append(config)
                    print(f"✓ PROFITABLE (ROI: {result['avg_roi']:.1f}%)")
                else:
                    if result:
                        print(f"✗ Not all profitable ({result['profitable_percentage']:.0f}% profitable)")
                    else:
                        print("✗ Analysis failed")
    
    elapsed = time.time() - start_time
    print(f"\nSearch completed in {elapsed:.1f} seconds")
    print(f"Tested {total_tested} configurations")
    print(f"Found {len(profitable_configs)} profitable configurations")
    
    if profitable_configs:
        print("\n📊 All Profitable Configurations:")
        print("-" * 90)
        print(f"{'Supply':>12} {'Price':>8} {'Airdrop%':>8} {'Market Cap':>15} {'Avg ROI':>10}")
        print("-" * 90)
        
        for config in sorted(profitable_configs, key=lambda x: x['avg_roi'], reverse=True):
            print(f"{config['supply']:>12,} ${config['price']:>7.2f} "
                  f"{config['airdrop_percent']:>7.0f}% ${config['market_cap']:>14,.0f} "
                  f"{config['avg_roi']:>9.1f}%")
        
        # Save to JSON
        output_file = Path("./working_profitable_configs.json")
        with open(output_file, 'w') as f:
            json.dump({
                'constraints': {
                    'min_price': 0.50,
                    'min_airdrop_percent': 40,
                    'opportunity_cost': 5,
                    'volatility': 30,
                    'certainty': 100
                },
                'search_summary': {
                    'total_tested': total_tested,
                    'profitable_found': len(profitable_configs)
                },
                'profitable_configurations': profitable_configs
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Generate CLI commands for top 5 configurations
        print(f"\n🚀 CLI Commands for All Profitable Configurations:")
        print("=" * 80)
        
        for i, config in enumerate(sorted(profitable_configs, key=lambda x: x['avg_roi'], reverse=True), 1):
            print(f"\n{i}. Market Cap ${config['market_cap']:,.0f} (ROI: {config['avg_roi']:.1f}%):")
            print(f"airdrop-calculator analyze \\")
            print(f"  --supply {config['supply']} \\")
            print(f"  --airdrop-percent {config['airdrop_percent']} \\")
            print(f"  --price {config['price']} \\")
            print(f"  --opportunity-cost 5 \\")
            print(f"  --volatility 30 \\")
            print(f"  --certainty 100 \\")
            print(f"  --charts all \\")
            print(f"  --output ./analysis_config_{i}")
            
            # Show segment breakdown
            print(f"\n   Segment Profitability:")
            for segment in config['segments']:
                status = "✓" if segment['profitable'] else "✗"
                print(f"   {status} {segment['name']}: {segment['roi']:.1f}% ROI")
        
        print(f"\n🎯 Market Cap Range: ${min(c['market_cap'] for c in profitable_configs):,.0f} - "
              f"${max(c['market_cap'] for c in profitable_configs):,.0f}")
        print(f"📈 ROI Range: {min(c['avg_roi'] for c in profitable_configs):.1f}% - "
              f"{max(c['avg_roi'] for c in profitable_configs):.1f}%")
        
    else:
        print("\n❌ No configurations found meeting all requirements!")
        print("The constraints may be too restrictive. Consider:")
        print("- Lowering minimum airdrop percentage")
        print("- Increasing opportunity cost")
        print("- Adjusting volatility")
        print("- Reducing minimum price requirement")

if __name__ == "__main__":
    find_profitable_configs()