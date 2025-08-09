#!/usr/bin/env python3
"""Test the calculate_market_metrics method."""

from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver

def test_metrics():
    """Test metrics calculation."""
    print("Testing calculate_market_metrics...")
    
    solver = EnhancedZ3Solver()
    calculator = AirdropCalculator(solver)
    
    try:
        metrics = calculator.calculate_market_metrics()
        print(f"✓ Metrics calculated successfully")
        print(f"  Min market cap: ${metrics.min_market_cap:,.0f}")
        print(f"  Profitable users: {metrics.profitable_users_percent:.1f}%")
        print(f"  Avg ROI: {metrics.avg_roi:.1f}%")
    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metrics()