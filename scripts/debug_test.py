#!/usr/bin/env python3
"""Debug script to isolate the test timeout issue."""

import time
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import Scenario

def test_with_timeout():
    """Test the hanging scenario with timeout."""
    print("Starting test...")
    start_time = time.time()
    
    solver = EnhancedZ3Solver()
    calculator = AirdropCalculator(solver)
    
    scenario = Scenario(
        name="Test Scenario",
        description="A simple test scenario.",
        constraints={
            "min_airdrop_percent": 20,
            "max_airdrop_percent": 50,
            "min_supply": 100_000_000,
            "max_supply": 1_000_000_000,
        },
        objectives={
            "market_cap": {"target": 250_000_000, "weight": 1.0},
            "profitability": {"target": 70, "weight": 0.5},
        }
    )
    
    print(f"Running scenario at {time.time() - start_time:.2f}s...")
    
    # Add debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        result = calculator.run_scenario(scenario)
        print(f"Completed at {time.time() - start_time:.2f}s")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error at {time.time() - start_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_timeout()