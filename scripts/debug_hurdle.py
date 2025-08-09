#!/usr/bin/env python3
"""Debug hurdle rate calculation"""

from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.defaults import OPTIMIZED_DEFAULTS
from airdrop_calculator.types import AirdropParameters

# Create calculator with optimized defaults
params = AirdropParameters(**OPTIMIZED_DEFAULTS)
calc = AirdropCalculator(params)

# Test beta calculation
r = params.opportunity_cost / 100  # 8% -> 0.08
sigma = params.volatility / 100     # 60% -> 0.6

print(f"Opportunity cost: {params.opportunity_cost}%")
print(f"Volatility: {params.volatility}%")
print(f"r = {r}, sigma = {sigma}")

beta = calc.calculate_beta(r, sigma)
print(f"Beta: {beta}")

hurdle = calc.calculate_hurdle_rate(params.opportunity_cost, params.volatility)
print(f"Hurdle rate: {hurdle}")

# Test cliff value
capital = 10000
cliff = calc.calculate_cliff_value(
    capital, 
    params.opportunity_cost,
    params.volatility,
    params.campaign_duration,
    params.gas_cost,
    10  # num transactions
)
print(f"\nFor ${capital} capital:")
print(f"Cliff value: ${cliff:.2f}")
print(f"Min profitable tokens: {cliff / params.launch_price:.2f}")

# Calculate expected allocation
allocation = calc.estimate_user_allocation(capital)
print(f"Expected allocation: {allocation:.2f} tokens")
print(f"Value at launch: ${allocation * params.launch_price:.2f}")

# Calculate ROI
gross_value = allocation * params.launch_price
total_cost = capital + (params.gas_cost * 10)
roi = ((gross_value - total_cost) / total_cost) * 100
print(f"\nROI Calculation:")
print(f"Gross value: ${gross_value:.2f}")
print(f"Total cost: ${total_cost:.2f}")
print(f"ROI: {roi:.2f}%")