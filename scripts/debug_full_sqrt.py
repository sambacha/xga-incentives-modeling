#!/usr/bin/env python3
"""Test the full solver with and without square root constraints."""

import time
from z3 import *

def test_full_with_sqrt():
    """Test full solver with square root constraint."""
    print("Testing FULL solver WITH square root constraint...")
    start_time = time.time()
    
    opt = Optimize()
    
    # All variables from the actual solver
    total_supply = Real('total_supply')
    airdrop_percent = Real('airdrop_percent')
    launch_price = Real('launch_price')
    opportunity_cost = Real('opportunity_cost')
    volatility = Real('volatility')
    gas_cost = Real('gas_cost')
    campaign_duration = Real('campaign_duration')
    airdrop_certainty = Real('airdrop_certainty')
    
    # Add all constraints
    opt.add(total_supply >= 100_000_000, total_supply <= 1_000_000_000)
    opt.add(airdrop_percent >= 20, airdrop_percent <= 50)
    opt.add(launch_price >= 0.01, launch_price <= 10.0)
    opt.add(opportunity_cost >= 2, opportunity_cost <= 50)
    opt.add(volatility >= 30, volatility <= 150)
    opt.add(gas_cost >= 10, gas_cost <= 500)
    opt.add(campaign_duration >= 3, campaign_duration <= 24)
    opt.add(airdrop_certainty >= 50, airdrop_certainty <= 100)
    
    # Add the problematic square root constraint
    r = opportunity_cost / 100
    sigma = volatility / 100
    sigma_squared = sigma * sigma
    
    a = 0.5 - r / sigma_squared
    discriminant_val = a * a + 2 * r / sigma_squared
    
    sqrt_discriminant = Real('sqrt_discriminant')
    opt.add(sqrt_discriminant * sqrt_discriminant == discriminant_val)
    opt.add(sqrt_discriminant >= 0)
    
    # Add objectives
    market_cap = total_supply * launch_price
    target_market_cap = 250_000_000
    market_cap_diff = If(market_cap > target_market_cap, 
                         market_cap - target_market_cap, 
                         target_market_cap - market_cap)
    market_cap_penalty = 1.0 * market_cap_diff / target_market_cap
    
    profitability = (30 - opportunity_cost) * 2 + airdrop_percent
    target_profitability = 70
    profitability_diff = If(profitability > target_profitability, 
                           profitability - target_profitability, 
                           target_profitability - profitability)
    profitability_penalty = 0.5 * profitability_diff / 100
    
    total_penalty = market_cap_penalty + profitability_penalty
    opt.minimize(total_penalty)
    
    print(f"All constraints added at {time.time() - start_time:.2f}s")
    
    opt.set("timeout", 10000)  # 10 second timeout
    
    check_start = time.time()
    result = opt.check()
    check_time = time.time() - check_start
    
    print(f"Solver completed in {check_time:.2f}s with result: {result}")
    
    return check_time

def test_full_without_sqrt():
    """Test full solver without square root constraint."""
    print("\nTesting FULL solver WITHOUT square root constraint...")
    start_time = time.time()
    
    opt = Optimize()
    
    # All variables from the actual solver
    total_supply = Real('total_supply')
    airdrop_percent = Real('airdrop_percent')
    launch_price = Real('launch_price')
    opportunity_cost = Real('opportunity_cost')
    volatility = Real('volatility')
    gas_cost = Real('gas_cost')
    campaign_duration = Real('campaign_duration')
    airdrop_certainty = Real('airdrop_certainty')
    
    # Add all constraints
    opt.add(total_supply >= 100_000_000, total_supply <= 1_000_000_000)
    opt.add(airdrop_percent >= 20, airdrop_percent <= 50)
    opt.add(launch_price >= 0.01, launch_price <= 10.0)
    opt.add(opportunity_cost >= 2, opportunity_cost <= 50)
    opt.add(volatility >= 30, volatility <= 150)
    opt.add(gas_cost >= 10, gas_cost <= 500)
    opt.add(campaign_duration >= 3, campaign_duration <= 24)
    opt.add(airdrop_certainty >= 50, airdrop_certainty <= 100)
    
    # Skip the square root constraint!
    
    # Add objectives
    market_cap = total_supply * launch_price
    target_market_cap = 250_000_000
    market_cap_diff = If(market_cap > target_market_cap, 
                         market_cap - target_market_cap, 
                         target_market_cap - market_cap)
    market_cap_penalty = 1.0 * market_cap_diff / target_market_cap
    
    profitability = (30 - opportunity_cost) * 2 + airdrop_percent
    target_profitability = 70
    profitability_diff = If(profitability > target_profitability, 
                           profitability - target_profitability, 
                           target_profitability - profitability)
    profitability_penalty = 0.5 * profitability_diff / 100
    
    total_penalty = market_cap_penalty + profitability_penalty
    opt.minimize(total_penalty)
    
    print(f"All constraints added at {time.time() - start_time:.2f}s")
    
    opt.set("timeout", 10000)  # 10 second timeout
    
    check_start = time.time()
    result = opt.check()
    check_time = time.time() - check_start
    
    print(f"Solver completed in {check_time:.2f}s with result: {result}")
    
    return check_time

if __name__ == "__main__":
    time_with = test_full_with_sqrt()
    time_without = test_full_without_sqrt()
    
    print(f"\nPerformance comparison:")
    print(f"  With sqrt constraint: {time_with:.2f}s")
    print(f"  Without sqrt constraint: {time_without:.2f}s")
    print(f"  Slowdown factor: {time_with/time_without:.1f}x")