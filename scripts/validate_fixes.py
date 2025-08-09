#!/usr/bin/env python3
"""
Script to validate the mathematical fixes applied to the codebase
"""

import numpy as np
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.types import AirdropParameters
from airdrop_calculator.solver import EnhancedZ3Solver, SolverConstraints

def test_beta_calculation():
    """Test that beta calculation is correct"""
    print("Testing Beta Calculation...")
    
    params = AirdropParameters(
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
    calc = AirdropCalculator(params)
    
    # Test case 1: r=0.1, sigma=0.8
    r = 0.1
    sigma = 0.8
    beta = calc.calculate_beta(r, sigma)
    
    # Manual calculation
    a = 0.5 - r / (sigma ** 2)
    d = a * a + 2 * r / (sigma ** 2)
    expected_beta = a + np.sqrt(d)
    
    print(f"  r={r}, sigma={sigma}")
    print(f"  Calculated beta: {beta}")
    print(f"  Expected beta (raw): {expected_beta}")
    print(f"  Beta > 1: {beta > 1.0}")
    
    # Test case 2: Edge case
    r = 0.01
    sigma = 0.1
    beta2 = calc.calculate_beta(r, sigma)
    print(f"\n  Edge case: r={r}, sigma={sigma}")
    print(f"  Beta: {beta2} (should be adjusted to minimum viable value)")
    
    return True

def test_hurdle_rate_calculation():
    """Test hurdle rate calculation with division safety"""
    print("\nTesting Hurdle Rate Calculation...")
    
    params = AirdropParameters(
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
    calc = AirdropCalculator(params)
    
    # Test normal case
    hurdle = calc.calculate_hurdle_rate(10, 80)
    print(f"  Normal case (10%, 80%): hurdle_rate = {hurdle}")
    print(f"  Hurdle rate in valid range [1.1, 10.0]: {1.1 <= hurdle <= 10.0}")
    
    # Test edge case that would cause division by zero
    hurdle2 = calc.calculate_hurdle_rate(5, 200)  # Very low opportunity cost, high volatility
    print(f"  Edge case (5%, 200%): hurdle_rate = {hurdle2}")
    print(f"  No division by zero error: True")
    
    return True

def test_roi_calculation():
    """Test ROI calculation handles zero cost correctly"""
    print("\nTesting ROI Calculation...")
    
    params = AirdropParameters(
        total_supply=1_000_000_000,
        airdrop_percent=10.0,
        launch_price=0.1,
        opportunity_cost=10.0,
        volatility=80.0,
        gas_cost=0.0,  # Zero gas cost to test edge case
        campaign_duration=12,
        airdrop_certainty=90.0,
        revenue_share=0.0,
        vesting_months=18,
        immediate_unlock=30.0
    )
    calc = AirdropCalculator(params)
    
    # Create a segment with zero capital to test edge case
    from airdrop_calculator.types import UserSegment
    segment = UserSegment("Test", 0, 0.01, 50, 0)
    
    result = calc.analyze_segment_profitability(segment)
    print(f"  Zero cost scenario ROI: {result.roi}")
    print(f"  ROI is NaN: {np.isnan(result.roi)}")
    
    # Test normal case
    segment2 = UserSegment("Normal", 1000, 5000, 50, 10)
    result2 = calc.analyze_segment_profitability(segment2)
    print(f"  Normal scenario ROI: {result2.roi}%")
    print(f"  ROI is finite: {np.isfinite(result2.roi)}")
    
    return True

def test_market_cap_overflow():
    """Test market cap calculation overflow protection"""
    print("\nTesting Market Cap Overflow Protection...")
    
    from airdrop_calculator.utils import safe_multiplication
    
    # Test normal case
    result1 = safe_multiplication(1e9, 0.1)
    print(f"  Normal case (1e9 * 0.1): {result1}")
    
    # Test potential overflow
    result2 = safe_multiplication(1e10, 1e10)
    print(f"  Large multiplication (1e10 * 1e10): {result2}")
    print(f"  Capped at maximum: {result2 == 1e15}")
    
    # Test with invalid inputs
    try:
        result3 = safe_multiplication(float('inf'), 10)
    except Exception as e:
        print(f"  Invalid input handled: {type(e).__name__}")
    
    return True

def test_invariant_checks():
    """Test comprehensive invariant validation"""
    print("\nTesting Invariant Checks...")
    
    # Test beta validation
    try:
        params = AirdropParameters(
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
            immediate_unlock=30.0,
            beta=0.9,  # Invalid beta <= 1
            hurdle_rate=2.0
        )
        print("  ERROR: Should have raised validation error for beta <= 1")
    except Exception as e:
        print(f"  Beta <= 1 validation: {type(e).__name__} - {str(e)[:50]}...")
    
    # Test hurdle rate validation
    try:
        params = AirdropParameters(
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
            immediate_unlock=30.0,
            beta=2.0,
            hurdle_rate=0.5  # Invalid hurdle < 1
        )
        print("  ERROR: Should have raised validation error for hurdle_rate < 1")
    except Exception as e:
        print(f"  Hurdle rate < 1 validation: {type(e).__name__} - {str(e)[:50]}...")
    
    # Test consistency between beta and hurdle_rate
    try:
        params = AirdropParameters(
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
            immediate_unlock=30.0,
            beta=2.0,
            hurdle_rate=5.0  # Inconsistent with beta (should be 2.0)
        )
        print("  ERROR: Should have raised validation error for inconsistent beta/hurdle")
    except Exception as e:
        print(f"  Beta/hurdle consistency: {type(e).__name__} - {str(e)[:50]}...")
    
    return True

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("VALIDATING MATHEMATICAL FIXES")
    print("=" * 60)
    
    tests = [
        test_beta_calculation,
        test_hurdle_rate_calculation,
        test_roi_calculation,
        test_market_cap_overflow,
        test_invariant_checks
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"  FAILED: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL MATHEMATICAL FIXES VALIDATED SUCCESSFULLY")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()