#!/usr/bin/env python3
"""Debug script to analyze ROI calculations"""

from airdrop_calculator.types import (
    AirdropParameters, TrackType, TrackParameters,
    NodeOperatorParameters
)
from airdrop_calculator.core import AirdropCalculator

# Default parameters from test_tracks.py
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

# Node operator example
node_track = TrackParameters(
    track_type=TrackType.NODE_OPERATOR,
    node_operator=NodeOperatorParameters(
        validators_operated=5,
        operation_duration_months=12,
        validator_performance_score=0.95,
        uptime_percentage=99.5
    )
)

calculator = AirdropCalculator(airdrop_params)
result = calculator.calculate_track_allocation(node_track)

print("=== Detailed ROI Breakdown ===")
print(f"\nAirdrop Parameters:")
print(f"  Total Supply: {airdrop_params.total_supply:,}")
print(f"  Airdrop %: {airdrop_params.airdrop_percent}%")
print(f"  Total Airdrop Tokens: {airdrop_params.total_supply * airdrop_params.airdrop_percent / 100:,}")
print(f"  Launch Price: ${airdrop_params.launch_price}")
print(f"  Gas Cost: ${airdrop_params.gas_cost}")
print(f"  Opportunity Cost: {airdrop_params.opportunity_cost}%")
print(f"  Volatility: {airdrop_params.volatility}%")

print(f"\nTrack Calculation:")
print(f"  Points: {result['points']:,.2f}")
print(f"  Capital Equivalent: ${result['capital_equivalent']:,.2f}")
print(f"  Allocation Model: {result['allocation_model']}")
print(f"  Estimated Allocation: {result['estimated_allocation']:,.2f} tokens")

print(f"\nProfitability Analysis:")
print(f"  Min Profitable Tokens: {result['min_profitable_tokens']:,.2f}")
print(f"  Is Profitable: {result['is_profitable']}")

print(f"\nROI Calculation:")
print(f"  Gross Value: ${result['gross_value']:,.2f} ({result['estimated_allocation']:,.2f} tokens × ${airdrop_params.launch_price})")
print(f"  Total Cost: ${result['total_cost']:,.2f} (${result['capital_equivalent']:,.2f} capital + ${airdrop_params.gas_cost * 10} gas)")
print(f"  Net Value: ${result['gross_value'] - result['total_cost']:,.2f}")
print(f"  ROI: {result['roi']:.2f}%")

# Calculate hurdle rate
hurdle_rate = calculator.calculate_hurdle_rate(airdrop_params.opportunity_cost, airdrop_params.volatility)
print(f"\nHurdle Rate Analysis:")
print(f"  Hurdle Rate Multiple: {hurdle_rate:.2f}x")
print(f"  Required Return: ${result['capital_equivalent'] * hurdle_rate:,.2f}")

# Test with different parameters
print("\n\n=== Testing Different Parameter Scenarios ===")

scenarios = [
    ("Higher Launch Price", {"launch_price": 1.0}),
    ("Lower Capital", {"launch_price": 0.1}),  # Reset to original
    ("Higher Airdrop %", {"airdrop_percent": 20.0}),
    ("Lower Opportunity Cost", {"opportunity_cost": 5.0}),
    ("Lower Volatility", {"volatility": 40.0}),
]

for scenario_name, changes in scenarios:
    # Create new params with changes
    params_dict = {
        "total_supply": airdrop_params.total_supply,
        "airdrop_percent": airdrop_params.airdrop_percent,
        "launch_price": airdrop_params.launch_price,
        "opportunity_cost": airdrop_params.opportunity_cost,
        "volatility": airdrop_params.volatility,
        "gas_cost": airdrop_params.gas_cost,
        "campaign_duration": airdrop_params.campaign_duration,
        "airdrop_certainty": airdrop_params.airdrop_certainty,
        "revenue_share": airdrop_params.revenue_share,
        "vesting_months": airdrop_params.vesting_months,
        "immediate_unlock": airdrop_params.immediate_unlock,
    }
    params_dict.update(changes)
    
    test_params = AirdropParameters(**params_dict)
    test_calc = AirdropCalculator(test_params)
    test_result = test_calc.calculate_track_allocation(node_track, test_params)
    
    print(f"\n{scenario_name}:")
    print(f"  Changes: {changes}")
    print(f"  Allocation: {test_result['estimated_allocation']:,.2f} tokens")
    print(f"  Gross Value: ${test_result['gross_value']:,.2f}")
    print(f"  ROI: {test_result['roi']:.2f}%")