#!/usr/bin/env python3
"""Test the performance impact of square root constraints."""

import time
from z3 import *

def test_with_sqrt_constraint():
    """Test solver with square root constraint."""
    print("Testing WITH square root constraint...")
    start_time = time.time()
    
    opt = Optimize()
    
    # Variables
    opportunity_cost = Real('opportunity_cost')
    volatility = Real('volatility')
    
    # Add bounds
    opt.add(opportunity_cost >= 2, opportunity_cost <= 50)
    opt.add(volatility >= 30, volatility <= 150)
    
    # Create the problematic square root constraint
    r = opportunity_cost / 100
    sigma = volatility / 100
    sigma_squared = sigma * sigma
    
    a = 0.5 - r / sigma_squared
    discriminant_val = a * a + 2 * r / sigma_squared
    
    sqrt_discriminant = Real('sqrt_discriminant')
    
    # This is the problematic constraint
    opt.add(sqrt_discriminant * sqrt_discriminant == discriminant_val)
    opt.add(sqrt_discriminant >= 0)
    
    # Simple objective
    opt.minimize(opportunity_cost)
    
    print(f"Constraints added at {time.time() - start_time:.2f}s")
    
    opt.set("timeout", 5000)  # 5 second timeout
    
    check_start = time.time()
    result = opt.check()
    check_time = time.time() - check_start
    
    print(f"Solver completed in {check_time:.2f}s with result: {result}")
    
    if result == sat:
        model = opt.model()
        print(f"Solution: opportunity_cost={model[opportunity_cost]}, volatility={model[volatility]}")

def test_without_sqrt_constraint():
    """Test solver without square root constraint."""
    print("\nTesting WITHOUT square root constraint...")
    start_time = time.time()
    
    opt = Optimize()
    
    # Variables
    opportunity_cost = Real('opportunity_cost')
    volatility = Real('volatility')
    
    # Add bounds
    opt.add(opportunity_cost >= 2, opportunity_cost <= 50)
    opt.add(volatility >= 30, volatility <= 150)
    
    # Simple objective
    opt.minimize(opportunity_cost)
    
    print(f"Constraints added at {time.time() - start_time:.2f}s")
    
    check_start = time.time()
    result = opt.check()
    check_time = time.time() - check_start
    
    print(f"Solver completed in {check_time:.2f}s with result: {result}")
    
    if result == sat:
        model = opt.model()
        print(f"Solution: opportunity_cost={model[opportunity_cost]}, volatility={model[volatility]}")

if __name__ == "__main__":
    test_with_sqrt_constraint()
    test_without_sqrt_constraint()