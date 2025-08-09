#!/usr/bin/env python3
"""
Test script for track functionality
"""

from airdrop_calculator.types import (
    AirdropParameters, TrackType, TrackParameters,
    NodeOperatorParameters, RiskUnderwriterParameters,
    LiquidityProviderParameters, AuctionParticipantParameters
)
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.tracks import MultiTrackCalculator

def test_single_tracks():
    """Test individual track calculations"""
    print("Testing Single Track Calculations")
    print("=" * 60)
    
    # Create optimized airdrop parameters
    from airdrop_calculator.defaults import OPTIMIZED_DEFAULTS
    airdrop_params = AirdropParameters(**OPTIMIZED_DEFAULTS)
    
    calculator = AirdropCalculator(airdrop_params)
    
    # Test Node Operator Track
    print("\n1. Node Operator Track:")
    node_track = TrackParameters(
        track_type=TrackType.NODE_OPERATOR,
        node_operator=NodeOperatorParameters(
            validators_operated=5,
            operation_duration_months=12,
            validator_performance_score=0.95,
            uptime_percentage=99.5
        )
    )
    result = calculator.calculate_track_allocation(node_track)
    print(f"   Points: {result['points']:,.2f}")
    print(f"   Capital Equivalent: ${result['capital_equivalent']:,.2f}")
    print(f"   Estimated Allocation: {result['estimated_allocation']:,.2f} tokens")
    print(f"   ROI: {result['roi']:.2f}%")
    
    # Test Risk Underwriter Track
    print("\n2. Risk Underwriter Track:")
    risk_track = TrackParameters(
        track_type=TrackType.RISK_UNDERWRITER,
        risk_underwriter=RiskUnderwriterParameters(
            tokens_staked=10000,
            staking_duration_months=24,
            token_type="FOLD"
        )
    )
    result = calculator.calculate_track_allocation(risk_track)
    print(f"   Points: {result['points']:,.2f}")
    print(f"   Capital Equivalent: ${result['capital_equivalent']:,.2f}")
    print(f"   Estimated Allocation: {result['estimated_allocation']:,.2f} tokens")
    print(f"   ROI: {result['roi']:.2f}%")
    
    # Test Liquidity Provider Track
    print("\n3. Liquidity Provider Track:")
    lp_track = TrackParameters(
        track_type=TrackType.LIQUIDITY_PROVIDER,
        liquidity_provider=LiquidityProviderParameters(
            lst_amount=5.0,
            liquidity_duration_months=12,
            pool_type="stETH-FOLD",
            pool_bonus_multiplier=1.2
        )
    )
    result = calculator.calculate_track_allocation(lp_track)
    print(f"   Points: {result['points']:,.2f}")
    print(f"   Capital Equivalent: ${result['capital_equivalent']:,.2f}")
    print(f"   Estimated Allocation: {result['estimated_allocation']:,.2f} tokens")
    print(f"   ROI: {result['roi']:.2f}%")
    
    # Test Auction Participant Track
    print("\n4. Auction Participant Track:")
    auction_track = TrackParameters(
        track_type=TrackType.AUCTION_PARTICIPANT,
        auction_participant=AuctionParticipantParameters(
            total_bids=50,
            successful_bids=40,
            bid_accuracy=85.0,
            total_bid_value=50000
        )
    )
    result = calculator.calculate_track_allocation(auction_track)
    print(f"   Points: {result['points']:,.2f}")
    print(f"   Capital Equivalent: ${result['capital_equivalent']:,.2f}")
    print(f"   Estimated Allocation: {result['estimated_allocation']:,.2f} tokens")
    print(f"   ROI: {result['roi']:.2f}%")

def test_multi_track():
    """Test multi-track calculations"""
    print("\n\nTesting Multi-Track Calculations")
    print("=" * 60)
    
    # Create optimized airdrop parameters
    from airdrop_calculator.defaults import OPTIMIZED_DEFAULTS
    airdrop_params = AirdropParameters(**OPTIMIZED_DEFAULTS)
    
    # Create multiple track parameters
    tracks = [
        TrackParameters(
            track_type=TrackType.NODE_OPERATOR,
            node_operator=NodeOperatorParameters(
                validators_operated=2,
                operation_duration_months=12,
                validator_performance_score=1.0,
                uptime_percentage=99.8
            )
        ),
        TrackParameters(
            track_type=TrackType.RISK_UNDERWRITER,
            risk_underwriter=RiskUnderwriterParameters(
                tokens_staked=5000,
                staking_duration_months=12,
                token_type="FOLD"
            )
        ),
        TrackParameters(
            track_type=TrackType.LIQUIDITY_PROVIDER,
            liquidity_provider=LiquidityProviderParameters(
                lst_amount=3.0,
                liquidity_duration_months=12,
                pool_type="default",
                pool_bonus_multiplier=1.0
            )
        )
    ]
    
    calculator = AirdropCalculator(airdrop_params)
    result = calculator.calculate_multi_track_allocation(tracks)
    
    print("\nTrack Summary:")
    for track_name, track_data in result['track_summary'].items():
        print(f"\n{track_name}:")
        print(f"  Points: {track_data['points']:,.2f}")
        print(f"  Capital Equivalent: ${track_data['capital_equivalent']:,.2f}")
        print(f"  Risk Factor: {track_data['risk_factor']:.2f}x")
    
    print(f"\nTotal Points: {result['total_points']:,.2f}")
    print(f"Total Capital Equivalent: ${result['total_capital_equivalent']:,.2f}")
    print(f"Weighted Risk Factor: {result['weighted_risk_factor']:.2f}x")
    print(f"\nRecommended Total Allocation: {result['recommended_allocation']:,.2f} tokens")
    print(f"Overall ROI: {result['overall_roi']:.2f}%")
    
    print("\nAllocation Breakdown:")
    for track_type, allocation in result['allocation_breakdown'].items():
        print(f"  {track_type.name}: {allocation:,.2f} tokens")

def test_strategy_comparison():
    """Test strategy comparison"""
    print("\n\nTesting Strategy Comparison")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'Conservative Node Focus',
            'tracks': [
                TrackParameters(
                    track_type=TrackType.NODE_OPERATOR,
                    node_operator=NodeOperatorParameters(
                        validators_operated=3,
                        operation_duration_months=24,
                        validator_performance_score=1.0,
                        uptime_percentage=99.9
                    )
                )
            ]
        },
        {
            'name': 'Balanced Strategy',
            'tracks': [
                TrackParameters(
                    track_type=TrackType.NODE_OPERATOR,
                    node_operator=NodeOperatorParameters(
                        validators_operated=1,
                        operation_duration_months=12,
                        validator_performance_score=1.0,
                        uptime_percentage=99.5
                    )
                ),
                TrackParameters(
                    track_type=TrackType.RISK_UNDERWRITER,
                    risk_underwriter=RiskUnderwriterParameters(
                        tokens_staked=10000,
                        staking_duration_months=12,
                        token_type="FOLD"
                    )
                )
            ]
        },
        {
            'name': 'Aggressive Multi-Track',
            'tracks': [
                TrackParameters(
                    track_type=TrackType.RISK_UNDERWRITER,
                    risk_underwriter=RiskUnderwriterParameters(
                        tokens_staked=50000,
                        staking_duration_months=36,
                        token_type="EIGEN"
                    )
                ),
                TrackParameters(
                    track_type=TrackType.AUCTION_PARTICIPANT,
                    auction_participant=AuctionParticipantParameters(
                        total_bids=100,
                        successful_bids=80,
                        bid_accuracy=90.0,
                        total_bid_value=100000
                    )
                )
            ]
        }
    ]
    
    # Optimized airdrop params
    from airdrop_calculator.defaults import OPTIMIZED_DEFAULTS
    default_params = AirdropParameters(**OPTIMIZED_DEFAULTS)
    
    calculator = AirdropCalculator(default_params)
    comparison = calculator.compare_track_strategies(scenarios)
    
    print(comparison['comparison_summary'])
    
    if comparison['best_scenario']:
        print(f"\nBest Strategy: {comparison['best_scenario']['scenario_name']}")

if __name__ == "__main__":
    try:
        test_single_tracks()
        test_multi_track()
        test_strategy_comparison()
        print("\n\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()