#!/usr/bin/env python3
"""Debug beta and hurdle rate calculation"""

import numpy as np

def calculate_beta_raw(r, sigma, delta=0.0):
    """Raw beta calculation without bounds"""
    sigma_squared = sigma ** 2
    a = 0.5 - (r - delta) / sigma_squared
    discriminant = a**2 + 2 * r / sigma_squared
    beta = a + np.sqrt(discriminant)
    return beta, a, discriminant

def calculate_hurdle_raw(beta):
    """Raw hurdle calculation"""
    return beta / (beta - 1)

# Test different parameter combinations
print("=== Beta and Hurdle Rate Analysis ===")
print("\nFormula: β = a + √(a² + 2r/σ²) where a = 0.5 - (r-δ)/σ²")
print("Hurdle Rate = β/(β-1)")

test_cases = [
    ("Default (10%, 80%)", 0.10, 0.80),
    ("Low vol (10%, 40%)", 0.10, 0.40),
    ("High vol (10%, 120%)", 0.10, 1.20),
    ("Low opp cost (5%, 80%)", 0.05, 0.80),
    ("High opp cost (15%, 80%)", 0.15, 0.80),
    ("Moderate (7%, 60%)", 0.07, 0.60),
    ("Conservative (5%, 40%)", 0.05, 0.40),
]

print("\n{:<25} {:>8} {:>8} {:>8} {:>12} {:>12}".format(
    "Scenario", "r (%)", "σ (%)", "β", "Hurdle", "Capped?"))
print("-" * 80)

for name, r, sigma in test_cases:
    beta, a, disc = calculate_beta_raw(r, sigma)
    hurdle = calculate_hurdle_raw(beta)
    
    # Check if it would be capped
    capped_beta = max(1.01, beta)
    capped_hurdle = max(1.1, min(hurdle, 10.0))
    is_capped = (capped_beta != beta) or (capped_hurdle != hurdle)
    
    print("{:<25} {:>8.1%} {:>8.1%} {:>8.4f} {:>12.2f} {:>12}".format(
        name, r, sigma, beta, hurdle, "Yes" if is_capped else "No"))

# Detailed calculation for default case
print("\n\n=== Detailed Calculation for Default Case (10%, 80%) ===")
r = 0.10
sigma = 0.80
sigma_squared = sigma ** 2

print(f"r = {r}")
print(f"σ = {sigma}")
print(f"σ² = {sigma_squared}")

a = 0.5 - r / sigma_squared
print(f"\na = 0.5 - r/σ² = 0.5 - {r}/{sigma_squared} = {a:.6f}")

disc = a**2 + 2 * r / sigma_squared
print(f"Discriminant = a² + 2r/σ² = {a:.6f}² + 2×{r}/{sigma_squared}")
print(f"            = {a**2:.6f} + {2*r/sigma_squared:.6f} = {disc:.6f}")

beta = a + np.sqrt(disc)
print(f"\nβ = a + √(discriminant) = {a:.6f} + √{disc:.6f}")
print(f"  = {a:.6f} + {np.sqrt(disc):.6f} = {beta:.6f}")

hurdle = beta / (beta - 1)
print(f"\nHurdle Rate = β/(β-1) = {beta:.6f}/({beta:.6f}-1)")
print(f"            = {beta:.6f}/{beta-1:.6f} = {hurdle:.2f}")

# The problem!
print(f"\n⚠️  ISSUE DETECTED:")
print(f"Beta = {beta:.6f} is very close to 1.0")
print(f"This makes (β-1) = {beta-1:.6f} very small")
print(f"Leading to hurdle rate = {hurdle:.2f} which is extremely high!")
print(f"The code caps this at 10.0 to prevent unrealistic values")

# What parameters would give reasonable hurdle rates?
print("\n\n=== Finding Reasonable Parameters ===")
print("Target hurdle rates between 1.5x and 3x")
print("\n{:<30} {:>8} {:>8} {:>8} {:>12}".format(
    "Parameters", "r (%)", "σ (%)", "β", "Hurdle"))
print("-" * 70)

for r_test in [0.05, 0.10, 0.15, 0.20]:
    for sigma_test in [0.20, 0.30, 0.40, 0.50]:
        beta_test, _, _ = calculate_beta_raw(r_test, sigma_test)
        hurdle_test = calculate_hurdle_raw(beta_test)
        if 1.5 <= hurdle_test <= 3.0:
            print("{:<30} {:>8.1%} {:>8.1%} {:>8.4f} {:>12.2f}".format(
                f"r={r_test:.0%}, σ={sigma_test:.0%}", 
                r_test, sigma_test, beta_test, hurdle_test))