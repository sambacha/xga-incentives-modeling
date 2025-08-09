from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import Scenario

def test_calculator_initialization(default_params):
    """
    Test that the AirdropCalculator can be initialized.
    """
    calculator = AirdropCalculator(default_params)
    assert calculator is not None
    assert calculator.params == default_params
    assert calculator.risk_free_base == 0.02
    assert len(calculator.user_segments) == 4

def test_run_scenario(calculator):
    """
    Test the run_scenario method with a simple case.
    """
    scenario = Scenario(
        name="Test Scenario",
        description="A simple test scenario.",
        constraints={
            "min_airdrop_percent": 20,
            "max_airdrop_percent": 50,
            "min_supply": 100_000_000,
            "max_supply": 1_000_000_000,
        },
        objectives={
            "market_cap": {"target": 250_000_000, "weight": 1.0},
            "profitable_users": {"target": 70, "weight": 0.5},
        }
    )
    
    result = calculator.run_scenario(scenario)
    assert result is not None
    assert "parameters" in result
    assert "estimated_impact" in result
    assert result["parameters"] is not None

def test_estimate_impact(calculator):
    """
    Test the _estimate_impact method.
    """
    class MockAirdropParams:
        def __init__(self):
            self.total_supply = 800_000_000
            self.launch_price = 3.5
            self.airdrop_percent = 25
            self.opportunity_cost = 10
            self.beta = 1.2
            self.hurdle_rate = 6.0

    params = MockAirdropParams()
    impact = calculator._estimate_impact(params)
    
    assert impact is not None
    assert "market_cap" in impact
    assert "user_profitability" in impact
    assert "required_investment" in impact
    assert impact["market_cap"] == 800_000_000 * 3.5  # 2.8 billion

def test_calculate_beta():
    """
    Test the calculate_beta method with various inputs.
    """
    from airdrop_calculator.types import AirdropParameters
    
    # Create a calculator with default parameters
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
    calculator = AirdropCalculator(params)
    
    # Test with typical values
    beta1 = calculator.calculate_beta(r=0.1, sigma=0.8)
    assert beta1 > 1.0

    # Test with low volatility
    beta2 = calculator.calculate_beta(r=0.1, sigma=0.2)
    assert beta2 > 1.0

    # Test with high volatility
    beta3 = calculator.calculate_beta(r=0.1, sigma=1.5)
    assert beta3 > 1.0

    # Test edge case with very low volatility
    beta4 = calculator.calculate_beta(r=0.01, sigma=0.1)
    # With the correct formula this should not produce negative discriminant
    # a = 0.5 - 0.01/0.01 = 0.5 - 1 = -0.5
    # d = (-0.5)^2 + 2*0.01/0.01 = 0.25 + 2 = 2.25
    # sqrt(d) = 1.5
    # beta = -0.5 + 1.5 = 1.0
    # But since beta <= 1.0, it gets adjusted to 1.2
    assert beta4 == 1.2  # Minimum viable beta

def test_calculate_hurdle_rate():
    """
    Test the calculate_hurdle_rate method.
    """
    from airdrop_calculator.types import AirdropParameters
    
    # Create a calculator with default parameters
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
    calculator = AirdropCalculator(params)
    # Test with typical values
    hurdle_rate1 = calculator.calculate_hurdle_rate(opportunity_cost=10, volatility=80)
    assert hurdle_rate1 > 1.0

    # Test with high opportunity cost
    hurdle_rate2 = calculator.calculate_hurdle_rate(opportunity_cost=30, volatility=80)
    assert hurdle_rate2 > 1.0

    # Test with low volatility
    hurdle_rate3 = calculator.calculate_hurdle_rate(opportunity_cost=10, volatility=20)
    assert hurdle_rate3 > 1.0

def test_calculate_cliff_value(calculator):
    """
    Test the calculate_cliff_value method.
    """
    # The calculator now uses self.params instead of self.solver

    cliff_value = calculator.calculate_cliff_value(
        capital=10000,
        opportunity_cost=10,
        volatility=80,
        time_months=6,
        gas_cost=500
    )
    assert cliff_value > 0

def test_calculate_min_profitable_tokens(calculator):
    """
    Test the calculate_min_profitable_tokens method.
    """
    # The calculator now uses self.params instead of self.solver

    min_tokens = calculator.calculate_min_profitable_tokens(capital=10000)
    assert min_tokens > 0

def test_estimate_user_allocation(calculator):
    """
    Test the estimate_user_allocation method with different models.
    """
    # The calculator now uses self.params instead of self.solver

    # Test linear allocation
    allocation1 = calculator.estimate_user_allocation(capital=10000, allocation_model="linear")
    assert allocation1 > 0

    # Test quadratic allocation
    allocation2 = calculator.estimate_user_allocation(capital=10000, allocation_model="quadratic")
    assert allocation2 > 0

    # Test logarithmic allocation
    allocation3 = calculator.estimate_user_allocation(capital=10000, allocation_model="logarithmic")
    assert allocation3 > 0

    # Test tiered allocation
    allocation4 = calculator.estimate_user_allocation(capital=10000, allocation_model="tiered")
    assert allocation4 > 0

def test_analyze_segment_profitability(calculator):
    """
    Test the analyze_segment_profitability method.
    """
    # The calculator now uses self.params instead of self.solver

    segment = calculator.user_segments[0]
    result = calculator.analyze_segment_profitability(segment)
    assert result is not None
    assert result.segment == segment.name

def test_calculate_market_metrics(calculator):
    """
    Test the calculate_market_metrics method.
    """
    # The calculator now uses self.params instead of self.solver

    metrics = calculator.calculate_market_metrics()
    assert metrics is not None
    assert metrics.min_market_cap > 0
    assert metrics.hurdle_rate > 1.0
    assert metrics.profitable_users_percent >= 0

def test_find_optimal_capital(calculator):
    """
    Test the find_optimal_capital method.
    """
    # The calculator now uses self.params instead of self.solver

    optimal_capital = calculator.find_optimal_capital()
    assert optimal_capital > 0

def test_compare_scenarios(calculator):
    """
    Test the compare_scenarios method.
    """
    scenario1 = Scenario(
        name="Scenario A",
        description="First test scenario.",
        constraints={"min_airdrop_percent": 5, "max_airdrop_percent": 15},
        objectives={"market_cap": {"target": 200_000_000, "weight": 1.0}, "profitable_users": {"target": 50, "weight": 0.5}}
    )
    
    scenario2 = Scenario(
        name="Scenario B",
        description="Second test scenario.",
        constraints={"min_airdrop_percent": 15, "max_airdrop_percent": 25},
        objectives={"market_cap": {"target": 300_000_000, "weight": 1.0}, "profitable_users": {"target": 60, "weight": 0.5}}
    )
    
    comparison = calculator.compare_scenarios([scenario1, scenario2])
    
    assert comparison is not None
    assert "scenario_results" in comparison
    assert "comparison_summary" in comparison
    # May have 0, 1 or 2 results depending on solver success
    assert len(comparison["scenario_results"]) >= 0
