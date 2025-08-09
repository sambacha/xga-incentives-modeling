"""
Pytest fixtures for airdrop calculator tests
"""
import pytest
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import AirdropParameters


@pytest.fixture
def default_params():
    """Default AirdropParameters for testing"""
    return AirdropParameters(
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


@pytest.fixture
def calculator(default_params):
    """AirdropCalculator instance with default parameters"""
    calc = AirdropCalculator(default_params)
    # Add solver for methods that need it
    calc.solver = EnhancedZ3Solver()
    return calc


@pytest.fixture
def solver():
    """EnhancedZ3Solver instance"""
    return EnhancedZ3Solver()