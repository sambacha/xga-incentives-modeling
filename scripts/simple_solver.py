#!/usr/bin/env python3
"""
Simple Airdrop Parameter Solver

Easy-to-use script for finding airdrop parameters with your specific constraints.
"""

import json
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

def solve_airdrop(
    opportunity_cost=5.0,
    min_airdrop_percent=30.0,
    min_price=0.1,
    gas_cost=1.0,
    min_duration=3,
    target_market_cap=500_000_000,
    save_to_file=True
):
    """
    Simple function to solve for airdrop parameters.
    
    Args:
        opportunity_cost: User opportunity cost percentage (default: 5.0)
        min_airdrop_percent: Minimum airdrop percentage (default: 30.0)
        min_price: Minimum launch price (default: 0.1)
        gas_cost: Gas cost per transaction (default: 1.0)
        min_duration: Minimum campaign duration in months (default: 3)
        target_market_cap: Target market capitalization (default: 500M)
        save_to_file: Whether to save results to JSON (default: True)
    """
    
    print("🔍 Solving for airdrop parameters...")
    print(f"Constraints: opportunity_cost={opportunity_cost}%, min_airdrop={min_airdrop_percent}%, min_price=${min_price}")
    
    solver = EnhancedZ3Solver()
    
    # Define constraints
    constraints = SolverConstraints(
        min_airdrop_percent=min_airdrop_percent,
        max_airdrop_percent=50.0,
        opportunity_cost=opportunity_cost,
        gas_cost=gas_cost,
        campaign_duration=min_duration,
        min_price=min_price,
        max_price=20.0,
        min_supply=10_000_000,
        max_supply=5_000_000_000
    )
    
    # Try Kalman solver first (most robust)
    print("Trying Kalman-enhanced solver...")
    solution = solver.solve_incremental_with_kalman(
        target_market_cap=target_market_cap,
        target_profitable_users=70.0,
        initial_constraints=constraints,
        max_iterations=10
    )
    
    if not solution:
        print("Trying soft constraints solver...")
        objectives = {
            'market_cap': (target_market_cap, 1.0),
            'profitable_users': (70.0, 0.5)
        }
        solution = solver.solve_with_soft_constraints(objectives, constraints)
    
    if solution:
        print("✅ Solution found!")
        print(f"Supply: {solution.total_supply:,.0f} tokens")
        print(f"Airdrop: {solution.airdrop_percent:.1f}%")
        print(f"Price: ${solution.launch_price:.4f}")
        print(f"Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
        print(f"Beta: {solution.beta:.4f}")
        print(f"Hurdle Rate: {solution.hurdle_rate:.4f}")
        
        if save_to_file:
            result = {
                "total_supply": solution.total_supply,
                "airdrop_percent": solution.airdrop_percent,
                "launch_price": solution.launch_price,
                "market_cap": solution.total_supply * solution.launch_price,
                "beta": solution.beta,
                "hurdle_rate": solution.hurdle_rate,
                "opportunity_cost": solution.opportunity_cost,
                "volatility": solution.volatility,
                "gas_cost": solution.gas_cost,
                "campaign_duration": solution.campaign_duration
            }
            
            with open("simple_solution.json", "w") as f:
                json.dump(result, f, indent=2)
            print("💾 Saved to simple_solution.json")
        
        return solution
    else:
        print("❌ No solution found. Try relaxing constraints.")
        return None

if __name__ == "__main__":
    # Your specific constraints
    solution = solve_airdrop(
        opportunity_cost=5.0,
        min_airdrop_percent=30.0,
        min_price=0.1,
        gas_cost=1.0,
        min_duration=3,
        target_market_cap=500_000_000
    )