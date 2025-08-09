#!/usr/bin/env python3
"""
Quick configuration finder for profitable airdrop scenarios.
Focused search with the specified constraints.
"""

import json
import time
from pathlib import Path
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import AirdropParameters

def find_profitable_configs():
    """Find configurations where all users are profitable with given constraints."""
    
    print("🔍 Quick Profitable Configuration Finder")
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
    
    # Fixed parameters
    fixed_params = {
        'opportunity_cost': 2.0,
        'volatility': 50.0,
        'airdrop_certainty': 100.0,
        'revenue_share': 40.0,
        'vesting_months': 12,
        'immediate_unlock': 30.0,
        'gas_cost': 50.0,
        'campaign_duration': 6
    }
    
    # Search ranges (smaller for quick testing)
    supplies = [100_000_000, 500_000_000, 1_000_000_000, 2_500_000_000]
    prices = [0.50, 1.00, 2.00, 5.00]
    airdrop_percents = [40, 50, 60, 80]
    
    profitable_configs = []
    total_tested = 0
    
    print("Searching configurations...")
    start_time = time.time()
    
    for supply in supplies:
        for price in prices:
            for airdrop_pct in airdrop_percents:
                total_tested += 1
                
                try:
                    # Create parameters
                    params = AirdropParameters(
                        total_supply=supply,
                        airdrop_percent=airdrop_pct,
                        launch_price=price,
                        **fixed_params
                    )
                    
                    # Calculate metrics
                    metrics = calculator.calculate_market_metrics()
                    market_cap = supply * price
                    
                    # Check if all users are profitable (>95% to account for rounding)
                    if metrics.profitable_users_percent >= 95.0:
                        
                        # Verify individual segments
                        all_segments_profitable = all(
                            sr.profitable and sr.roi > 0 
                            for sr in metrics.segment_results
                        )
                        
                        if all_segments_profitable:
                            config = {
                                'supply': supply,
                                'price': price,
                                'airdrop_percent': airdrop_pct,
                                'market_cap': market_cap,
                                'profitable_users_pct': metrics.profitable_users_percent,
                                'avg_roi': metrics.avg_roi,
                                'segments': [
                                    {
                                        'name': sr.segment,
                                        'roi': sr.roi,
                                        'profitable': sr.profitable
                                    }
                                    for sr in metrics.segment_results
                                ]
                            }
                            profitable_configs.append(config)
                            
                            print(f"✓ Found: {supply:,} tokens × ${price} = ${market_cap:,.0f} "
                                  f"({airdrop_pct}% airdrop, ROI: {metrics.avg_roi:.1f}%)")
                
                except Exception as e:
                    print(f"✗ Error with {supply:,} × ${price}: {e}")
                    continue
    
    elapsed = time.time() - start_time
    print(f"\nSearch completed in {elapsed:.1f} seconds")
    print(f"Tested {total_tested} configurations")
    print(f"Found {len(profitable_configs)} profitable configurations")
    
    if profitable_configs:
        print("\n📊 All Profitable Configurations:")
        print("-" * 80)
        print(f"{'Supply':>12} {'Price':>8} {'Airdrop%':>8} {'Market Cap':>12} {'Avg ROI':>8}")
        print("-" * 80)
        
        for config in sorted(profitable_configs, key=lambda x: x['avg_roi'], reverse=True):
            print(f"{config['supply']:>12,} ${config['price']:>7.2f} "
                  f"{config['airdrop_percent']:>7.0f}% ${config['market_cap']:>11,.0f} "
                  f"{config['avg_roi']:>7.1f}%")
        
        # Save to JSON
        output_file = Path("./quick_profitable_configs.json")
        with open(output_file, 'w') as f:
            json.dump({
                'search_summary': {
                    'total_tested': total_tested,
                    'profitable_found': len(profitable_configs),
                    'constraints': fixed_params
                },
                'profitable_configurations': profitable_configs
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Generate CLI commands for top 5 configurations
        print(f"\n🚀 CLI Commands for Top 5 Configurations:")
        print("-" * 50)
        
        for i, config in enumerate(sorted(profitable_configs, key=lambda x: x['avg_roi'], reverse=True)[:5], 1):
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
        
    else:
        print("\n❌ No configurations found meeting all requirements!")
        print("The constraints may be too restrictive. Consider:")
        print("- Lowering minimum airdrop percentage")
        print("- Increasing opportunity cost")
        print("- Adjusting volatility")

if __name__ == "__main__":
    find_profitable_configs()
