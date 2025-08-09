#!/usr/bin/env python3
"""Test complete workflow with track calculations and charts"""

import json
from pathlib import Path
from airdrop_calculator.types import (
    AirdropParameters, TrackType, TrackParameters,
    NodeOperatorParameters, RiskUnderwriterParameters,
    LiquidityProviderParameters, AuctionParticipantParameters
)
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.visualization import AirdropVisualizer
from airdrop_calculator.defaults import OPTIMIZED_DEFAULTS

# Create optimized parameters
params = AirdropParameters(**OPTIMIZED_DEFAULTS)
calculator = AirdropCalculator(params)

# Define track parameters
tracks = [
    TrackParameters(
        track_type=TrackType.NODE_OPERATOR,
        node_operator=NodeOperatorParameters(
            validators_operated=5,
            operation_duration_months=12,
            validator_performance_score=0.95,
            uptime_percentage=99.5
        )
    ),
    TrackParameters(
        track_type=TrackType.RISK_UNDERWRITER,
        risk_underwriter=RiskUnderwriterParameters(
            tokens_staked=10000,
            staking_duration_months=24,
            token_type="FOLD"
        )
    ),
    TrackParameters(
        track_type=TrackType.LIQUIDITY_PROVIDER,
        liquidity_provider=LiquidityProviderParameters(
            lst_amount=5.0,
            liquidity_duration_months=12,
            pool_type="stETH-FOLD",
            pool_bonus_multiplier=1.2
        )
    ),
    TrackParameters(
        track_type=TrackType.AUCTION_PARTICIPANT,
        auction_participant=AuctionParticipantParameters(
            total_bids=50,
            successful_bids=40,
            bid_accuracy=85.0,
            total_bid_value=50000
        )
    )
]

# Calculate track allocations
print("Calculating track allocations...")
track_results = []
for track in tracks:
    result = calculator.calculate_track_allocation(track)
    track_results.append(result)
    print(f"  {result['track_type']}: {result['estimated_allocation']:,.0f} tokens (ROI: {result['roi']:.1f}%)")

# Store track results in calculator for visualization
calculator.track_results = track_results

# Create visualizer
visualizer = AirdropVisualizer(calculator)

# Generate all charts including track dashboard
print("\nGenerating all charts including track analysis...")
output_dir = "complete_test_charts"
results = visualizer.generate_all_charts(output_dir)

print("\nChart generation results:")
for chart_type, path in results.items():
    status = "✓" if path else "✗"
    print(f"{status} {chart_type}: {path}")

# Test multi-track calculation
print("\nTesting multi-track calculation...")
multi_result = calculator.calculate_multi_track_allocation(tracks)
print(f"Total allocation: {multi_result['recommended_allocation']:,.0f} tokens")
print(f"Overall ROI: {multi_result['overall_roi']:.1f}%")

# Generate just the track dashboard with multi-track results
print("\nGenerating standalone track dashboard...")
from airdrop_calculator.track_visualizations import TrackVisualizer
track_viz = TrackVisualizer()

# Create combined results for visualization
combined_results = track_results + [
    {
        'track_type': 'MULTI_TRACK',
        'points': multi_result['total_points'],
        'capital_equivalent': multi_result['total_capital_equivalent'],
        'estimated_allocation': multi_result['recommended_allocation'],
        'risk_factor': multi_result['weighted_risk_factor'],
        'roi': multi_result['overall_roi']
    }
]

track_viz.plot_track_performance_dashboard(
    combined_results,
    save_path="complete_test_track_dashboard.png"
)
print("✓ Standalone track dashboard saved")

# Save summary report
summary = {
    "parameters": OPTIMIZED_DEFAULTS,
    "track_results": track_results,
    "multi_track_result": {
        "total_allocation": float(multi_result['recommended_allocation']),
        "overall_roi": float(multi_result['overall_roi']),
        "allocation_breakdown": {k.name: float(v) for k, v in multi_result['allocation_breakdown'].items()}
    },
    "charts_generated": list(results.keys())
}

with open("complete_test_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n✓ Summary saved to complete_test_summary.json")

print("\nComplete workflow test finished successfully!")