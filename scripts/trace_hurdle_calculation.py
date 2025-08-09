#!/usr/bin/env python3
"""Trace through the hurdle rate calculation to understand the issue"""

import numpy as np

# Simulate the calculation with the current implementation
def calculate_beta_current(r, sigma, delta=0.0):
    """Current implementation from the code"""
    # Bounds checking
    r = max(0.001, min(r, 0.5))
    sigma = max(0.1, min(sigma, 2.0))
    delta = max(0, min(delta, r))
    
    sigma_squared = sigma ** 2
    a = 0.5 - (r - delta) / sigma_squared
    
    discriminant = a**2 + 2 * r / sigma_squared
    beta = a + np.sqrt(discriminant)
    
    # The problematic line - forces beta to be at least 1.01
    return max(1.01, beta)

def calculate_hurdle_current(opportunity_cost, volatility):
    """Current hurdle calculation"""
    r = opportunity_cost / 100
    sigma = volatility / 100
    
    beta = calculate_beta_current(r, sigma)
    
    # Add epsilon to prevent division by zero
    epsilon = 1e-9
    hurdle = beta / (beta - 1 + epsilon)
    
    # Bound hurdle rate
    return max(1.1, min(hurdle, 10.0))

print("=== Tracing Hurdle Rate Calculation ===")

# Test with default parameters
opp_cost = 10.0  # 10%
volatility = 80.0  # 80%

print(f"\nInput Parameters:")
print(f"  Opportunity Cost: {opp_cost}%")
print(f"  Volatility: {volatility}%")

r = opp_cost / 100
sigma = volatility / 100

print(f"\nConverted to decimals:")
print(f"  r = {r}")
print(f"  σ = {sigma}")

# Calculate beta step by step
sigma_squared = sigma ** 2
a = 0.5 - r / sigma_squared
discriminant = a**2 + 2 * r / sigma_squared
beta_raw = a + np.sqrt(discriminant)

print(f"\nBeta Calculation:")
print(f"  σ² = {sigma_squared}")
print(f"  a = 0.5 - r/σ² = 0.5 - {r}/{sigma_squared} = {a}")
print(f"  discriminant = a² + 2r/σ² = {a}² + 2×{r}/{sigma_squared} = {discriminant}")
print(f"  β (raw) = a + √(discriminant) = {a} + {np.sqrt(discriminant)} = {beta_raw}")

# The issue
beta_capped = max(1.01, beta_raw)
print(f"\n⚠️  Beta Capping:")
print(f"  β (capped) = max(1.01, {beta_raw}) = {beta_capped}")

# Calculate hurdle
epsilon = 1e-9
hurdle_raw = beta_capped / (beta_capped - 1 + epsilon)
hurdle_capped = max(1.1, min(hurdle_raw, 10.0))

print(f"\nHurdle Calculation:")
print(f"  Hurdle (raw) = β/(β-1) = {beta_capped}/({beta_capped}-1+{epsilon}) = {hurdle_raw}")
print(f"  Hurdle (capped) = max(1.1, min({hurdle_raw}, 10.0)) = {hurdle_capped}")

print("\n\n=== The Problem ===")
print(f"When β = 1.0, it gets capped to 1.01")
print(f"This makes hurdle = 1.01/(1.01-1) = 1.01/0.01 = 101")
print(f"Which then gets capped to 10.0")
print(f"\nThis means the hurdle rate is always 10x for many parameter combinations!")

# Test different volatility values
print("\n\n=== Testing Different Volatilities ===")
print("{:<15} {:>10} {:>10} {:>10} {:>12}".format(
    "Volatility", "β (raw)", "β (cap)", "Hurdle", "Final"))
print("-" * 60)

for vol in [20, 40, 60, 80, 100, 120, 150, 200]:
    r = 0.10
    sigma = vol / 100
    sigma_squared = sigma ** 2
    a = 0.5 - r / sigma_squared
    discriminant = a**2 + 2 * r / sigma_squared
    beta_raw = a + np.sqrt(discriminant)
    beta_capped = max(1.01, beta_raw)
    hurdle_raw = beta_capped / (beta_capped - 1 + 1e-9)
    hurdle_final = max(1.1, min(hurdle_raw, 10.0))
    
    print("{:<15} {:>10.4f} {:>10.4f} {:>10.2f} {:>12.2f}".format(
        f"{vol}%", beta_raw, beta_capped, hurdle_raw, hurdle_final))

# What would be reasonable parameters?
print("\n\n=== Suggested Parameter Ranges for Positive ROI ===")
print("\nTo achieve positive ROI, we need:")
print("1. Lower hurdle rate (currently always 10x)")
print("2. Higher token allocation")
print("3. Higher launch price")
print("4. Lower capital requirements")

# Calculate required changes
current_allocation = 336_794  # From the test output
current_price = 0.10
current_capital = 113_430
current_gas = 500
current_gross = current_allocation * current_price
current_cost = current_capital + current_gas

print(f"\nCurrent Situation:")
print(f"  Allocation: {current_allocation:,} tokens")
print(f"  Launch Price: ${current_price}")
print(f"  Gross Value: ${current_gross:,.2f}")
print(f"  Total Cost: ${current_cost:,.2f}")
print(f"  Required Return (10x): ${current_capital * 10:,.2f}")

print(f"\nTo achieve breakeven, need one of:")
print(f"  - Launch price: ${current_cost / current_allocation:.2f} (vs current ${current_price})")
print(f"  - Token allocation: {current_cost / current_price:,.0f} tokens (vs current {current_allocation:,})")
print(f"  - Hurdle rate: {current_gross / current_capital:.2f}x (vs current 10x)")

print(f"\nTo achieve 50% ROI, need:")
target_gross = current_cost * 1.5
print(f"  - Launch price: ${target_gross / current_allocation:.2f}")
print(f"  - Token allocation: {target_gross / current_price:,.0f} tokens")
print(f"  - Or reduce hurdle rate to ~{target_gross / current_capital:.2f}x")