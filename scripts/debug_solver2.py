#!/usr/bin/env python3
"""Debug solver issues"""

from airdrop_calculator.solver import EnhancedZ3Solver, SolverConstraints

def test_enhanced_solver():
    """Test the EnhancedZ3Solver directly"""
    print("Testing EnhancedZ3Solver...")
    
    solver = EnhancedZ3Solver()
    constraints = SolverConstraints(
        opportunity_cost=10.0,
        volatility=80.0,
        min_airdrop_percent=5,
        max_airdrop_percent=50
    )
    
    print("Trying soft constraints solver...")
    objectives = {'market_cap': (200_000_000, 1.0), 'profitable_users': (60, 1.0)}
    solution = solver.solve_with_soft_constraints(objectives, constraints)
    
    if solution:
        print(f"Solution found!")
        print(f"  Beta: {solution.beta}")
        print(f"  Hurdle rate: {solution.hurdle_rate}")
        print(f"  Supply: {solution.total_supply}")
        print(f"  Price: {solution.launch_price}")
        print(f"  Market cap: {solution.total_supply * solution.launch_price}")
        return True
    else:
        print("No solution found by soft constraints solver!")
        print("\nCalculating expected beta manually:")
        r_val = 0.1
        sigma_val = 0.8
        a_val = 0.5 - r_val / (sigma_val ** 2)
        d_val = a_val ** 2 + 2 * r_val / (sigma_val ** 2)
        beta_val = a_val + d_val ** 0.5
        print(f"  r = {r_val}, sigma = {sigma_val}")
        print(f"  a = {a_val}")
        print(f"  discriminant = {d_val}")
        print(f"  beta = {beta_val}")
        print(f"  Beta >= 1.2? {beta_val >= 1.2}")
        return False

if __name__ == "__main__":
    test_enhanced_solver()