import pytest
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.analysis import ProfitabilityAnalyzer
from airdrop_calculator.types import AirdropParameters

@pytest.fixture
def analyzer():
    """
    Fixture for the ProfitabilityAnalyzer.
    """
    solver = EnhancedZ3Solver()
    # The calculator needs default parameters to be initialized.
    default_params = AirdropParameters(
        total_supply=1_000_000_000, airdrop_percent=10, launch_price=1.0,
        opportunity_cost=10, volatility=80, gas_cost=500,
        campaign_duration=6, airdrop_certainty=70, revenue_share=10,
        vesting_months=12, immediate_unlock=25
    )
    calculator = AirdropCalculator(solver)
    return ProfitabilityAnalyzer(calculator)

def test_calculate_cliff_value(analyzer: ProfitabilityAnalyzer):
    """
    Test the calculate_cliff_value method.
    """
    result = analyzer.calculate_cliff_value(
        capital=10000,
        opportunity_cost_rate=0.1,
        time_months=6
    )
    assert result is not None
    assert "risk_adjusted_required_value" in result
    assert result["risk_adjusted_required_value"] > 0
    assert "hurdle_multiple" in result
    assert result["hurdle_multiple"] > 1

def test_analyze_strategy(analyzer: ProfitabilityAnalyzer):
    """
    Test the analyze_strategy method.
    """
    result = analyzer.analyze_strategy(
        strategy_name="Test Strategy",
        capital=10000,
        opportunity_cost_rate=0.1,
        expected_allocation_share=0.001,
        total_airdrop_tokens=100_000_000
    )
    assert result is not None
    assert result["strategy_name"] == "Test Strategy"
    assert "required_token_price" in result
    assert result["required_token_price"] > 0
    assert "required_market_cap" in result
    assert result["required_market_cap"] > 0
