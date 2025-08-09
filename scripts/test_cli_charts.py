#!/usr/bin/env python3
"""Test CLI chart generation with track data"""

import json
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.visualization import AirdropVisualizer
from airdrop_calculator.defaults import OPTIMIZED_DEFAULTS
from airdrop_calculator.types import AirdropParameters

# Create test data
params = AirdropParameters(**OPTIMIZED_DEFAULTS)
calculator = AirdropCalculator(params)

# Add track results to calculator for testing
calculator.track_results = [
    {
        'track_type': 'NODE_OPERATOR',
        'points': 56.71,
        'capital_equivalent': 56715.00,
        'estimated_allocation': 357223.67,
        'risk_factor': 0.8,
        'roi': -37.56
    },
    {
        'track_type': 'RISK_UNDERWRITER',
        'points': 30000.00,
        'capital_equivalent': 10000.00,
        'estimated_allocation': 15000.00,
        'risk_factor': 1.2,
        'roi': -85.71
    },
    {
        'track_type': 'LIQUIDITY_PROVIDER',
        'points': 6.00,
        'capital_equivalent': 15000.00,
        'estimated_allocation': 183711.73,
        'risk_factor': 1.0,
        'roi': 18.52
    },
    {
        'track_type': 'AUCTION_PARTICIPANT',
        'points': 34000.00,
        'capital_equivalent': 1680.00,
        'estimated_allocation': 6048.00,
        'risk_factor': 1.0,
        'roi': -72.26
    }
]

# Create visualizer
visualizer = AirdropVisualizer(calculator)

# Generate all charts
print("Generating all charts including track analysis...")
results = visualizer.generate_all_charts("test_charts_output")

print("\nChart generation results:")
for chart_type, path in results.items():
    status = "✓" if path else "✗"
    print(f"{status} {chart_type}: {path}")

# Check if track dashboard was generated
if 'track_dashboard' in results and results['track_dashboard']:
    print("\nTrack dashboard successfully integrated!")
else:
    print("\nWarning: Track dashboard was not generated")

# Save test config for CLI
test_config = {
    "total_supply": OPTIMIZED_DEFAULTS['total_supply'],
    "airdrop_percent": OPTIMIZED_DEFAULTS['airdrop_percent'],
    "launch_price": OPTIMIZED_DEFAULTS['launch_price'],
    "opportunity_cost": OPTIMIZED_DEFAULTS['opportunity_cost'],
    "volatility": OPTIMIZED_DEFAULTS['volatility'],
    "gas_cost": OPTIMIZED_DEFAULTS['gas_cost'],
    "campaign_duration": OPTIMIZED_DEFAULTS['campaign_duration'],
    "airdrop_certainty": OPTIMIZED_DEFAULTS['airdrop_certainty'],
    "revenue_share": OPTIMIZED_DEFAULTS['revenue_share'],
    "vesting_months": OPTIMIZED_DEFAULTS['vesting_months'],
    "immediate_unlock": OPTIMIZED_DEFAULTS['immediate_unlock']
}

with open("test_config.json", "w") as f:
    json.dump(test_config, f, indent=2)
print("\nTest config saved to test_config.json")