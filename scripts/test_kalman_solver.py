#!/usr/bin/env python3
"""
Test the Kalman filter-enhanced incremental solver.

This demonstrates how the Kalman filter learns optimal constraint relaxation
strategies to improve solving performance over time.
"""

import logging
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

# Set up logging to see the adaptive learning process
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_kalman_vs_traditional():
    """Compare Kalman-enhanced vs traditional incremental solving"""
    print("=" * 80)
    print("TESTING KALMAN FILTER-ENHANCED INCREMENTAL SOLVER")
    print("=" * 80)
    
    solver = EnhancedZ3Solver()
    
    # Define test scenario
    target_market_cap = 500_000_000  # $500M market cap
    target_profitable_users = 80     # 80% profitable users
    
    # Define initial constraints (somewhat restrictive)
    constraints = SolverConstraints(
        min_supply=800_000_000,
        max_supply=1_200_000_000,
        min_price=0.25,
        max_price=0.75,
        min_airdrop_percent=15,
        max_airdrop_percent=35
    )
    
    print(f"\nTarget: ${target_market_cap/1e6:.0f}M market cap, {target_profitable_users}% profitable users")
    print(f"Initial constraints: Supply {constraints.min_supply/1e6:.0f}M-{constraints.max_supply/1e6:.0f}M")
    print(f"                    Price ${constraints.min_price:.2f}-${constraints.max_price:.2f}")
    print(f"                    Airdrop {constraints.min_airdrop_percent}%-{constraints.max_airdrop_percent}%")
    
    # Test Kalman-enhanced approach
    print("\n" + "=" * 50)
    print("KALMAN FILTER-ENHANCED APPROACH")
    print("=" * 50)
    
    try:
        solution_kalman = solver.solve_incremental_with_kalman(
            target_market_cap, 
            target_profitable_users, 
            constraints,
            max_iterations=15
        )
        
        if solution_kalman:
            print(f"\n✓ Kalman solution found:")
            print(f"  Supply: {solution_kalman.total_supply:,.0f}")
            print(f"  Airdrop %: {solution_kalman.airdrop_percent:.1f}%")
            print(f"  Launch Price: ${solution_kalman.launch_price:.3f}")
            print(f"  Market Cap: ${solution_kalman.total_supply * solution_kalman.launch_price/1e6:.1f}M")
            print(f"  Opportunity Cost: {solution_kalman.opportunity_cost:.1f}%")
            print(f"  Volatility: {solution_kalman.volatility:.1f}%")
        else:
            print("✗ No Kalman solution found")
            
    except Exception as e:
        print(f"✗ Kalman approach failed: {e}")
    
    # Test traditional incremental approach for comparison
    print("\n" + "=" * 50)
    print("TRADITIONAL INCREMENTAL APPROACH")
    print("=" * 50)
    
    try:
        levels = [
            (100, constraints),
            (50, SolverConstraints(min_airdrop_percent=10, max_airdrop_percent=40)),
            (10, SolverConstraints())
        ]
        
        solution_traditional = solver.solve_incremental_with_relaxation(
            target_market_cap,
            target_profitable_users,
            levels
        )
        
        if solution_traditional:
            print(f"\n✓ Traditional solution found:")
            print(f"  Supply: {solution_traditional.total_supply:,.0f}")
            print(f"  Airdrop %: {solution_traditional.airdrop_percent:.1f}%")
            print(f"  Launch Price: ${solution_traditional.launch_price:.3f}")
            print(f"  Market Cap: ${solution_traditional.total_supply * solution_traditional.launch_price/1e6:.1f}M")
            print(f"  Opportunity Cost: {solution_traditional.opportunity_cost:.1f}%")
            print(f"  Volatility: {solution_traditional.volatility:.1f}%")
        else:
            print("✗ No traditional solution found")
            
    except Exception as e:
        print(f"✗ Traditional approach failed: {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
The Kalman filter approach offers several advantages:

1. ADAPTIVE LEARNING: Learns optimal constraint relaxation levels from previous attempts
2. PREDICTIVE TIMING: Estimates solve times to set appropriate timeouts
3. QUALITY TRACKING: Monitors solution quality to find the best solutions
4. EARLY TERMINATION: Stops when high-quality solutions are found
5. PERFORMANCE MONITORING: Tracks success rates and convergence patterns

The traditional approach uses fixed relaxation levels, while Kalman adapts
based on observed performance, leading to more efficient solving.
""")

def test_kalman_learning():
    """Demonstrate how Kalman filter learns over multiple solve attempts"""
    print("\n" + "=" * 80)
    print("DEMONSTRATING KALMAN FILTER LEARNING")
    print("=" * 80)
    
    solver = EnhancedZ3Solver()
    
    # Test with progressively challenging scenarios
    scenarios = [
        (100_000_000, 70, "Easy scenario"),      # $100M, 70% profitable
        (500_000_000, 80, "Medium scenario"),    # $500M, 80% profitable  
        (1_000_000_000, 85, "Hard scenario"),   # $1B, 85% profitable
        (2_000_000_000, 90, "Very hard scenario") # $2B, 90% profitable
    ]
    
    constraints = SolverConstraints(
        min_supply=500_000_000,
        max_supply=2_000_000_000,
        min_price=0.1,
        max_price=2.0,
        min_airdrop_percent=10,
        max_airdrop_percent=50
    )
    
    for i, (market_cap, profitable_users, description) in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {description} ---")
        print(f"Target: ${market_cap/1e6:.0f}M market cap, {profitable_users}% profitable")
        
        try:
            solution = solver.solve_incremental_with_kalman(
                market_cap,
                profitable_users,
                constraints,
                max_iterations=10
            )
            
            if solution:
                actual_cap = solution.total_supply * solution.launch_price
                print(f"✓ Solution: ${actual_cap/1e6:.1f}M market cap achieved")
            else:
                print("✗ No solution found")
                
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_kalman_vs_traditional()
    test_kalman_learning()