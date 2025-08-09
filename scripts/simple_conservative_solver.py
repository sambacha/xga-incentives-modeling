#!/usr/bin/env python3
"""
Simple Conservative Airdrop Solver

Easy-to-use script for conservative airdrop scenarios with low opportunity cost and volatility.
"""

import json
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

def solve_conservative_airdrop(
    opportunity_cost=0.5,
    min_airdrop_percent=30.0,
    min_price=0.1,
    revenue_share=40.0,
    volatility=20.0,
    gas_cost=1.0,
    min_duration=3,
    target_market_cap=500_000_000,
    save_to_file=True
):
    """
    Simple function to solve for conservative airdrop parameters.
    
    Args:
        opportunity_cost: User opportunity cost percentage (default: 0.5%)
        min_airdrop_percent: Minimum airdrop percentage (default: 30.0%)
        min_price: Minimum launch price (default: 0.1)
        revenue_share: Revenue share percentage (default: 40.0%)
        volatility: Market volatility percentage (default: 20.0%)
        gas_cost: Gas cost per transaction (default: 1.0)
        min_duration: Minimum campaign duration in months (default: 3)
        target_market_cap: Target market capitalization (default: 500M)
        save_to_file: Whether to save results to JSON (default: True)
    """
    
    print("🔍 Solving for CONSERVATIVE airdrop parameters...")
    print(f"Conservative constraints: opp_cost={opportunity_cost}%, volatility={volatility}%, min_airdrop={min_airdrop_percent}%")
    print(f"Revenue share: {revenue_share}%, min_price=${min_price}")
    
    solver = EnhancedZ3Solver()
    
    # Conservative constraints with flexibility for relaxation
    constraints = SolverConstraints(
        min_airdrop_percent=min_airdrop_percent,
        max_airdrop_percent=60.0,  # Higher flexibility
        opportunity_cost=opportunity_cost,
        volatility=volatility,
        gas_cost=gas_cost,
        campaign_duration=min_duration,
        min_price=min_price,
        max_price=15.0,  # Conservative price range
        min_supply=20_000_000,
        max_supply=30_000_000_000  # Large supply range for low prices
    )
    
    solution = None
    
    # Method 1: Try Kalman solver (best for conservative scenarios)
    print("Trying Kalman-enhanced solver...")
    solution = solver.solve_incremental_with_kalman(
        target_market_cap=target_market_cap,
        target_profitable_users=75.0,
        initial_constraints=constraints,
        max_iterations=15
    )
    
    # Method 2: Try soft constraints if Kalman fails
    if not solution:
        print("Trying soft constraints solver...")
        objectives = {
            'market_cap': (target_market_cap, 1.0),
            'profitable_users': (75.0, 0.5)
        }
        solution = solver.solve_with_soft_constraints(objectives, constraints)
    
    # Method 3: Try with relaxed constraints
    if not solution:
        print("Trying with relaxed constraints...")
        relaxed_constraints = SolverConstraints(
            min_airdrop_percent=max(20.0, min_airdrop_percent - 5),
            max_airdrop_percent=70.0,
            min_price=max(0.01, min_price - 0.05),
            max_price=25.0,
            min_supply=10_000_000,
            max_supply=50_000_000_000
        )
        
        solution = solver.solve_with_soft_constraints(
            objectives={'market_cap': (target_market_cap * 0.8, 1.0)},
            constraints=relaxed_constraints
        )
    
    if solution:
        # Set the revenue share (solver doesn't handle this parameter)
        solution.revenue_share = revenue_share
        
        print("✅ CONSERVATIVE SOLUTION FOUND!")
        print(f"Supply: {solution.total_supply:,.0f} tokens")
        print(f"Airdrop: {solution.airdrop_percent:.1f}%")
        print(f"Price: ${solution.launch_price:.4f}")
        print(f"Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
        print(f"Revenue Share: {solution.revenue_share:.1f}%")
        print(f"Opportunity Cost: {solution.opportunity_cost:.2f}%")
        print(f"Volatility: {solution.volatility:.1f}%")
        print(f"Beta: {solution.beta:.6f}")
        print(f"Hurdle Rate: {solution.hurdle_rate:.6f}")
        
        # Conservative scenario analysis
        print("\n📊 CONSERVATIVE ANALYSIS:")
        if solution.opportunity_cost <= 1.0:
            print(f"  ✅ Very low opportunity cost: {solution.opportunity_cost:.2f}% (Excellent retention)")
        elif solution.opportunity_cost <= 3.0:
            print(f"  ✓ Low opportunity cost: {solution.opportunity_cost:.2f}% (Good retention)")
        else:
            print(f"  ⚠️ Moderate opportunity cost: {solution.opportunity_cost:.2f}%")
        
        if solution.volatility <= 25.0:
            print(f"  ✅ Low volatility: {solution.volatility:.1f}% (Stable scenario)")
        elif solution.volatility <= 40.0:
            print(f"  ✓ Moderate volatility: {solution.volatility:.1f}% (Acceptable)")
        else:
            print(f"  ⚠️ Higher volatility: {solution.volatility:.1f}%")
        
        if solution.launch_price <= 1.0:
            print(f"  ✅ Very accessible price: ${solution.launch_price:.4f}")
        elif solution.launch_price <= 5.0:
            print(f"  ✓ Accessible price: ${solution.launch_price:.4f}")
        else:
            print(f"  ⚠️ Higher price point: ${solution.launch_price:.4f}")
        
        if solution.airdrop_percent >= 30.0:
            print(f"  ✅ Generous airdrop: {solution.airdrop_percent:.1f}%")
        elif solution.airdrop_percent >= 20.0:
            print(f"  ✓ Substantial airdrop: {solution.airdrop_percent:.1f}%")
        else:
            print(f"  ⚠️ Limited airdrop: {solution.airdrop_percent:.1f}%")
        
        if save_to_file:
            result = {
                "scenario": "conservative",
                "total_supply": solution.total_supply,
                "airdrop_percent": solution.airdrop_percent,
                "launch_price": solution.launch_price,
                "market_cap": solution.total_supply * solution.launch_price,
                "airdrop_tokens": solution.total_supply * (solution.airdrop_percent / 100),
                "revenue_share": solution.revenue_share,
                "opportunity_cost": solution.opportunity_cost,
                "volatility": solution.volatility,
                "beta": solution.beta,
                "hurdle_rate": solution.hurdle_rate,
                "gas_cost": solution.gas_cost,
                "campaign_duration": solution.campaign_duration,
                "conservative_metrics": {
                    "low_risk": solution.opportunity_cost <= 1.0 and solution.volatility <= 25.0,
                    "accessible": solution.launch_price <= 1.0,
                    "generous": solution.airdrop_percent >= 30.0,
                    "high_revenue_share": solution.revenue_share >= 30.0
                }
            }
            
            with open("conservative_simple_solution.json", "w") as f:
                json.dump(result, f, indent=2)
            print("💾 Saved to conservative_simple_solution.json")
        
        return solution
    else:
        print("❌ No conservative solution found.")
        print("Conservative scenarios with very low opportunity cost and volatility")
        print("may require adjusted targets or different mathematical approaches.")
        return None

if __name__ == "__main__":
    # Your specific conservative constraints
    solution = solve_conservative_airdrop(
        opportunity_cost=0.5,     # Very low
        min_airdrop_percent=30.0, # Generous
        min_price=0.1,           # Accessible
        revenue_share=40.0,      # High sustainability
        volatility=20.0,         # Low risk
        gas_cost=1.0,            # Low cost
        min_duration=3,          # Short campaign
        target_market_cap=500_000_000
    )