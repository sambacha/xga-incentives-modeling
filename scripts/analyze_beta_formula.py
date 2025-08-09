#!/usr/bin/env python3
"""Analyze the beta formula to understand when it equals 1"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_beta_components(r, sigma, delta=0.0):
    """Calculate beta and show all components"""
    sigma_squared = sigma ** 2
    a = 0.5 - (r - delta) / sigma_squared
    discriminant = a**2 + 2 * r / sigma_squared
    sqrt_term = np.sqrt(discriminant)
    beta = a + sqrt_term
    return {
        'r': r,
        'sigma': sigma,
        'sigma_squared': sigma_squared,
        'a': a,
        'discriminant': discriminant,
        'sqrt_term': sqrt_term,
        'beta': beta,
        'beta_minus_1': beta - 1
    }

print("=== Analysis of Beta Formula ===")
print("\nβ = a + √(a² + 2r/σ²) where a = 0.5 - (r-δ)/σ²")
print("\nFor β = 1, we need: a + √(a² + 2r/σ²) = 1")
print("Which means: √(a² + 2r/σ²) = 1 - a")

# Analyze the default case
print("\n\n=== Default Case (r=10%, σ=80%) ===")
result = calculate_beta_components(0.1, 0.8)
for key, value in result.items():
    print(f"{key}: {value:.6f}")

# Check the math
print("\n\nVerification:")
print(f"a + sqrt_term = {result['a']:.6f} + {result['sqrt_term']:.6f} = {result['beta']:.6f}")

# When does beta = 1?
print("\n\n=== When Beta Equals 1 ===")
print("β = 1 when: a + √(a² + 2r/σ²) = 1")
print("Rearranging: √(a² + 2r/σ²) = 1 - a")
print("Squaring both sides: a² + 2r/σ² = (1-a)²")
print("Expanding: a² + 2r/σ² = 1 - 2a + a²")
print("Simplifying: 2r/σ² = 1 - 2a")
print("Since a = 0.5 - r/σ²:")
print("2r/σ² = 1 - 2(0.5 - r/σ²)")
print("2r/σ² = 1 - 1 + 2r/σ²")
print("0 = 0")
print("\n⚠️  This is always true! The formula seems to have an issue.")

# Try the exact formula from financial literature
print("\n\n=== Checking Alternative Beta Formula ===")
print("For perpetual American options, the correct formula is:")
print("β = 0.5 - (r-δ)/σ² + √[(r-δ)/σ² - 0.5]² + 2r/σ²")

def calculate_beta_correct(r, sigma, delta=0.0):
    """Correct beta calculation for perpetual American options"""
    sigma_squared = sigma ** 2
    term1 = (r - delta) / sigma_squared
    inside_sqrt = (term1 - 0.5)**2 + 2*r/sigma_squared
    beta = 0.5 - term1 + np.sqrt(inside_sqrt)
    return beta

# Test with correct formula
print("\nResults with correct formula:")
test_params = [
    (0.10, 0.80),
    (0.10, 0.40),
    (0.05, 0.80),
    (0.15, 0.60),
]

for r, sigma in test_params:
    beta = calculate_beta_correct(r, sigma)
    hurdle = beta / (beta - 1) if beta != 1 else float('inf')
    print(f"r={r:.0%}, σ={sigma:.0%}: β={beta:.4f}, hurdle={hurdle:.2f}x")

# Plot beta surface
print("\n\nGenerating beta surface plot...")
r_range = np.linspace(0.02, 0.30, 50)
sigma_range = np.linspace(0.20, 1.50, 50)
R, S = np.meshgrid(r_range, sigma_range)

Beta = np.zeros_like(R)
for i in range(len(r_range)):
    for j in range(len(sigma_range)):
        Beta[j, i] = calculate_beta_correct(R[j, i], S[j, i])

plt.figure(figsize=(10, 8))
plt.contourf(R*100, S*100, Beta, levels=20, cmap='viridis')
plt.colorbar(label='Beta')
plt.xlabel('Opportunity Cost (%)')
plt.ylabel('Volatility (%)')
plt.title('Beta Surface for Perpetual American Options')
plt.contour(R*100, S*100, Beta, levels=[1.5, 2.0, 2.5, 3.0], colors='white', linewidths=2)
plt.savefig('beta_surface.png', dpi=150, bbox_inches='tight')
print("Saved beta surface plot to beta_surface.png")

# Find parameter combinations that give reasonable hurdle rates
print("\n\n=== Parameter Combinations for Reasonable Hurdle Rates ===")
print("Target: Hurdle rate between 1.2x and 3.0x")
print("\n{:<20} {:>10} {:>10} {:>10} {:>12}".format(
    "Parameters", "r (%)", "σ (%)", "β", "Hurdle"))
print("-" * 65)

good_params = []
for r in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    for sigma in [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
        beta = calculate_beta_correct(r, sigma)
        if beta > 1:
            hurdle = beta / (beta - 1)
            if 1.2 <= hurdle <= 3.0:
                good_params.append((r, sigma, beta, hurdle))
                print("{:<20} {:>10.1%} {:>10.1%} {:>10.4f} {:>12.2f}x".format(
                    f"r={r:.0%}, σ={sigma:.0%}", r, sigma, beta, hurdle))