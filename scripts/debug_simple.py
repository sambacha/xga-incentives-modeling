#!/usr/bin/env python3
"""Test with simpler objectives to isolate the issue."""

import time
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

def test_simple_solve():
    """Test with simpler objectives."""
    print("Testing simple solve...")
    start_time = time.time()
    
    solver = EnhancedZ3Solver()
    
    # Simple objectives without complex ratios
    objectives = {
        'total_supply': (500_000_000, 1.0),
        'airdrop_percent': (30, 1.0)
    }
    
    constraints = SolverConstraints()
    
    print(f"Starting solve at {time.time() - start_time:.2f}s...")
    
    result = solver.solve_with_soft_constraints(objectives, constraints)
    
    print(f"Completed at {time.time() - start_time:.2f}s")
    print(f"Result: {result}")
    
    if result:
        print(f"Solution found:")
        print(f"  total_supply: {result.total_supply}")
        print(f"  airdrop_percent: {result.airdrop_percent}")
        print(f"  beta: {result.beta}")
        print(f"  hurdle_rate: {result.hurdle_rate}")

if __name__ == "__main__":
    test_simple_solve()