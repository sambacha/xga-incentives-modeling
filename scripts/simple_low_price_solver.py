#!/usr/bin/env python3
"""
Simple Low Price Airdrop Solver

Easy-to-use script for finding airdrop parameters with strict price limits.
No Kalman solver, focuses on accessible pricing.
"""

import json
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

def solve_low_price_airdrop(
    opportunity_cost=0.5,
    min_airdrop_percent=30.0,
    min_price=0.1,
    max_price=0.50,
    revenue_share=40.0,
    volatility=20.0,
    gas_cost=1.0,
    min_duration=3,
    target_market_cap=500_000_000,
    save_to_file=True
):
    """
    Simple function to solve for low price airdrop parameters.
    
    Args:
        opportunity_cost: User opportunity cost percentage (default: 0.5%)
        min_airdrop_percent: Minimum airdrop percentage (default: 30.0%)
        min_price: Minimum launch price (default: 0.1)
        max_price: Maximum launch price - STRICT LIMIT (default: 0.50)
        revenue_share: Revenue share percentage (default: 40.0%)
        volatility: Market volatility percentage (default: 20.0%)
        gas_cost: Gas cost per transaction (default: 1.0)
        min_duration: Minimum campaign duration in months (default: 3)
        target_market_cap: Target market capitalization (default: 500M)
        save_to_file: Whether to save results to JSON (default: True)
    """
    
    print("🔍 Solving for LOW PRICE airdrop parameters...")
    print(f"STRICT PRICE LIMIT: ${min_price} - ${max_price}")
    print(f"Constraints: opp_cost={opportunity_cost}%, volatility={volatility}%, min_airdrop={min_airdrop_percent}%")
    print(f"Revenue share: {revenue_share}%")
    print("Using NON-KALMAN solvers only")
    
    solver = EnhancedZ3Solver()
    
    # Calculate supply range needed for price constraints
    min_supply_needed = target_market_cap / max_price
    max_supply_allowed = target_market_cap / min_price
    
    print(f"Supply range for target: {min_supply_needed:,.0f} - {max_supply_allowed:,.0f} tokens")
    
    # Initial constraints optimized for low price
    constraints = SolverConstraints(
        min_airdrop_percent=min_airdrop_percent,
        max_airdrop_percent=50.0,
        opportunity_cost=opportunity_cost,
        volatility=volatility,
        gas_cost=gas_cost,
        campaign_duration=min_duration,
        min_price=min_price,
        max_price=max_price,  # STRICT LIMIT
        min_supply=max(100_000_000, int(min_supply_needed * 0.8)),
        max_supply=min(50_000_000_000, int(max_supply_allowed * 1.2))
    )
    
    solution = None
    
    # Method 1: Soft constraints
    print("Trying soft constraints solver...")
    objectives = {
        'market_cap': (target_market_cap, 1.5),
        'profitable_users': (75.0, 0.7)
    }
    solution = solver.solve_with_soft_constraints(objectives, constraints)
    
    if solution and solution.launch_price <= max_price:
        print(f"✅ Found solution with soft constraints! Price: ${solution.launch_price:.6f}")
    else:
        solution = None
        print("❌ Soft constraints failed or price too high")
    
    # Method 2: Try with lower market cap targets
    if not solution:
        print("Trying with lower market cap targets...")
        for factor in [0.8, 0.6, 0.4, 0.2]:
            adjusted_target = target_market_cap * factor
            print(f"  Trying {factor*100:.0f}% of target: ${adjusted_target:,.0f}")
            
            # Adjust supply constraints for lower target
            adj_min_supply = int(adjusted_target / max_price)
            adj_max_supply = int(adjusted_target / min_price) * 2
            
            relaxed_constraints = SolverConstraints(
                min_airdrop_percent=max(10.0, min_airdrop_percent - 10),
                max_airdrop_percent=70.0,
                min_price=min_price,
                max_price=max_price,  # Keep strict
                min_supply=adj_min_supply,
                max_supply=adj_max_supply
            )
            
            objectives = {
                'market_cap': (adjusted_target, 1.0),
                'profitable_users': (60.0, 0.5)
            }
            
            solution = solver.solve_with_soft_constraints(objectives, relaxed_constraints)
            
            if solution and solution.launch_price <= max_price:
                print(f"✅ Found solution at {factor*100:.0f}% target! Price: ${solution.launch_price:.6f}")
                break
            else:
                solution = None
    
    # Method 3: Progressive relaxation maintaining price limit
    if not solution:
        print("Trying progressive relaxation with strict price limit...")
        
        relaxation_levels = [
            (80, SolverConstraints(
                min_airdrop_percent=20.0,
                max_airdrop_percent=60.0,
                min_price=min_price,
                max_price=max_price,
                min_supply=1_000_000_000,
                max_supply=20_000_000_000
            )),
            (60, SolverConstraints(
                min_airdrop_percent=15.0,
                max_airdrop_percent=70.0,
                min_price=min_price,
                max_price=max_price,
                min_supply=2_000_000_000,
                max_supply=50_000_000_000
            )),
            (40, SolverConstraints(
                min_airdrop_percent=10.0,
                max_airdrop_percent=80.0,
                min_price=min_price,
                max_price=max_price,
                min_supply=5_000_000_000,
                max_supply=100_000_000_000
            ))
        ]
        
        solution = solver.solve_incremental_with_relaxation(
            target_market_cap=target_market_cap * 0.3,  # 30% of target
            target_profitable_users=50.0,
            constraint_levels=relaxation_levels
        )
        
        if solution and solution.launch_price <= max_price:
            print(f"✅ Found solution with relaxation! Price: ${solution.launch_price:.6f}")
        else:
            solution = None
            print("❌ Progressive relaxation failed")
    
    if solution:
        # Set revenue share
        solution.revenue_share = revenue_share
        
        print("\n" + "="*60)
        print("🎉 LOW PRICE SOLUTION FOUND!")
        print("="*60)
        print(f"✅ Price: ${solution.launch_price:.6f} (within ${max_price} limit)")
        print(f"✅ Supply: {solution.total_supply:,.0f} tokens")
        print(f"✅ Airdrop: {solution.airdrop_percent:.1f}%")
        print(f"✅ Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
        print(f"✅ Revenue Share: {solution.revenue_share:.1f}%")
        print(f"✅ Airdrop Value: ${solution.total_supply * (solution.airdrop_percent/100) * solution.launch_price:,.0f}")
        
        # Accessibility analysis
        print(f"\n📊 ACCESSIBILITY ANALYSIS:")
        price_margin = ((max_price - solution.launch_price) / max_price) * 100
        print(f"  💰 Price margin: {price_margin:.1f}% below limit")
        print(f"  🎯 Tokens per $1: {1/solution.launch_price:.1f}")
        
        if solution.launch_price <= 0.1:
            print(f"  ✅ EXTREMELY accessible (mass market pricing)")
        elif solution.launch_price <= 0.3:
            print(f"  ✅ VERY accessible (broad retail)")
        else:
            print(f"  ✓ Accessible (retail friendly)")
        
        # Economic viability
        print(f"\n💡 ECONOMIC METRICS:")
        print(f"  Opportunity Cost: {solution.opportunity_cost:.2f}%")
        print(f"  Volatility: {solution.volatility:.1f}%")
        print(f"  Beta: {solution.beta:.6f}")
        print(f"  Hurdle Rate: {solution.hurdle_rate:.6f}")
        
        if save_to_file:
            result = {
                "scenario": "low_price",
                "price_limit_compliance": {
                    "launch_price": solution.launch_price,
                    "max_limit": max_price,
                    "within_limit": solution.launch_price <= max_price,
                    "margin_percent": price_margin
                },
                "core_parameters": {
                    "total_supply": solution.total_supply,
                    "airdrop_percent": solution.airdrop_percent,
                    "launch_price": solution.launch_price,
                    "market_cap": solution.total_supply * solution.launch_price,
                    "airdrop_tokens": solution.total_supply * (solution.airdrop_percent / 100),
                    "airdrop_value": solution.total_supply * (solution.airdrop_percent / 100) * solution.launch_price
                },
                "economic_parameters": {
                    "revenue_share": solution.revenue_share,
                    "opportunity_cost": solution.opportunity_cost,
                    "volatility": solution.volatility,
                    "beta": solution.beta,
                    "hurdle_rate": solution.hurdle_rate
                },
                "accessibility_metrics": {
                    "extremely_accessible": solution.launch_price <= 0.1,
                    "tokens_per_dollar": 1 / solution.launch_price,
                    "mass_market_pricing": solution.launch_price <= 0.1,
                    "retail_friendly": solution.launch_price <= 0.5
                }
            }
            
            with open("simple_low_price_solution.json", "w") as f:
                json.dump(result, f, indent=2)
            print("💾 Saved to simple_low_price_solution.json")
        
        print("="*60)
        return solution
    else:
        print("\n" + "="*60)
        print("❌ NO LOW PRICE SOLUTION FOUND")
        print("="*60)
        print(f"Unable to find solution under ${max_price} price limit")
        print("Suggestions:")
        print(f"  1. Increase max price above ${max_price}")
        print(f"  2. Lower target market cap below ${target_market_cap:,.0f}")
        print(f"  3. Reduce minimum airdrop percentage below {min_airdrop_percent}%")
        print(f"  4. Accept much higher token supply")
        print("="*60)
        return None

if __name__ == "__main__":
    # Your specific low price constraints
    solution = solve_low_price_airdrop(
        opportunity_cost=0.5,     # Very low
        min_airdrop_percent=30.0, # Generous
        min_price=0.1,           # Accessible minimum
        max_price=0.50,          # STRICT MAXIMUM
        revenue_share=40.0,      # High sustainability
        volatility=20.0,         # Low risk
        gas_cost=1.0,            # Low cost
        min_duration=3,          # Short campaign
        target_market_cap=500_000_000  # May need to adjust
    )