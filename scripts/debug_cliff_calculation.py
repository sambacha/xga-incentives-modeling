#!/usr/bin/env python3
"""Debug script to analyze cliff value calculation"""

from airdrop_calculator.types import AirdropParameters
from airdrop_calculator.core import AirdropCalculator

# Default parameters
airdrop_params = AirdropParameters(
    total_supply=1_000_000_000,
    airdrop_percent=10.0,
    launch_price=0.1,
    opportunity_cost=10.0,
    volatility=80.0,
    gas_cost=50.0,
    campaign_duration=12,
    airdrop_certainty=90.0,
    revenue_share=0.0,
    vesting_months=18,
    immediate_unlock=30.0
)

calculator = AirdropCalculator(airdrop_params)

# Test capital amount
capital = 113_430

# Calculate step by step
r = airdrop_params.opportunity_cost / 100  # 0.1
sigma = airdrop_params.volatility / 100    # 0.8
time_years = airdrop_params.campaign_duration / 12  # 1

print("=== Cliff Value Calculation Breakdown ===")
print(f"\nInputs:")
print(f"  Capital: ${capital:,.2f}")
print(f"  Opportunity Cost Rate (r): {r:.2%}")
print(f"  Volatility (σ): {sigma:.2%}")
print(f"  Time (years): {time_years}")
print(f"  Gas Cost: ${airdrop_params.gas_cost}")
print(f"  Certainty: {airdrop_params.airdrop_certainty}%")

# Calculate beta
beta = calculator.calculate_beta(r, sigma)
print(f"\nBeta Calculation:")
print(f"  σ² = {sigma**2:.4f}")
print(f"  a = 0.5 - (r-δ)/σ² = 0.5 - {r}/{sigma**2:.4f} = {0.5 - r/(sigma**2):.4f}")
print(f"  β = a + √(a² + 2r/σ²) = {beta:.4f}")

# Calculate hurdle rate
hurdle = calculator.calculate_hurdle_rate(airdrop_params.opportunity_cost, airdrop_params.volatility)
print(f"\nHurdle Rate:")
print(f"  Hurdle = β/(β-1) = {beta:.4f}/({beta:.4f}-1) = {hurdle:.4f}")

# Calculate components of cliff value
opp_cost = capital * r * time_years
tx_costs = airdrop_params.gas_cost * 10 * 1.2  # 20% safety margin
required_value = capital * hurdle + opp_cost + tx_costs
certainty_factor = airdrop_params.airdrop_certainty / 100
risk_adjusted = required_value / certainty_factor

print(f"\nCliff Value Components:")
print(f"  1. Opportunity Cost = ${capital:,.2f} × {r:.2%} × {time_years} = ${opp_cost:,.2f}")
print(f"  2. Transaction Costs = ${airdrop_params.gas_cost} × 10 × 1.2 = ${tx_costs:,.2f}")
print(f"  3. Required Value = ${capital:,.2f} × {hurdle:.2f} + ${opp_cost:,.2f} + ${tx_costs:,.2f}")
print(f"     = ${capital * hurdle:,.2f} + ${opp_cost:,.2f} + ${tx_costs:,.2f}")
print(f"     = ${required_value:,.2f}")
print(f"  4. Risk Adjusted = ${required_value:,.2f} / {certainty_factor:.2f} = ${risk_adjusted:,.2f}")

# Calculate minimum tokens
min_tokens = risk_adjusted / airdrop_params.launch_price
print(f"\nMinimum Profitable Tokens:")
print(f"  = Cliff Value / Launch Price")
print(f"  = ${risk_adjusted:,.2f} / ${airdrop_params.launch_price}")
print(f"  = {min_tokens:,.2f} tokens")

# Show what's needed for profitability
print(f"\n=== Profitability Requirements ===")
print(f"To be profitable, need either:")
print(f"  1. More allocation: {min_tokens:,.0f} tokens (currently getting 336,794)")
print(f"  2. Higher launch price: ${risk_adjusted / 336_794:.2f} (currently $0.10)")
print(f"  3. Lower hurdle rate via:")
print(f"     - Lower opportunity cost (currently {airdrop_params.opportunity_cost}%)")
print(f"     - Lower volatility (currently {airdrop_params.volatility}%)")

# Test parameter sensitivity
print(f"\n=== Parameter Sensitivity ===")
test_scenarios = [
    ("Opportunity Cost 5%", 5, airdrop_params.volatility),
    ("Opportunity Cost 15%", 15, airdrop_params.volatility),
    ("Volatility 40%", airdrop_params.opportunity_cost, 40),
    ("Volatility 120%", airdrop_params.opportunity_cost, 120),
]

for name, opp_cost, vol in test_scenarios:
    test_hurdle = calculator.calculate_hurdle_rate(opp_cost, vol)
    print(f"\n{name}:")
    print(f"  Hurdle Rate: {test_hurdle:.2f}x")
    print(f"  Required Return: ${capital * test_hurdle:,.2f}")