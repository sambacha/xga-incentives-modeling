from typing import Dict, List
from .core import AirdropCalculator
from .types import AirdropParameters

class ProfitabilityAnalyzer:
    """
    Performs risk-adjusted expected value analysis for retroactive airdrops.
    """
    def __init__(self, calculator: AirdropCalculator):
        self.calculator = calculator

    def calculate_cliff_value(
        self,
        capital: float,
        opportunity_cost_rate: float,
        time_months: int,
        airdrop_probability: float = 0.7,
        volatility: float = 80.0,
        gas_cost: float = 500.0
    ) -> Dict:
        """
        Calculates the minimum required value for an airdrop to be profitable.
        """
        hurdle_multiple = self.calculator.calculate_hurdle_rate(
            opportunity_cost=opportunity_cost_rate * 100,
            volatility=volatility
        )

        opportunity_cost = capital * opportunity_cost_rate * (time_months / 12)
        required_value = (capital * hurdle_multiple) + opportunity_cost + gas_cost
        risk_adjusted_value = required_value / airdrop_probability

        return {
            "risk_adjusted_required_value": risk_adjusted_value,
            "hurdle_multiple": hurdle_multiple,
            "opportunity_cost": opportunity_cost,
        }

    def analyze_strategy(
        self,
        strategy_name: str,
        capital: float,
        opportunity_cost_rate: float,
        expected_allocation_share: float,
        total_airdrop_tokens: float,
        time_months: int = 6,
    ) -> Dict:
        """
        Analyzes a specific profitability scenario, like 'Liquidity Provider' or 'Whale'.
        """
        cliff_data = self.calculate_cliff_value(
            capital=capital,
            opportunity_cost_rate=opportunity_cost_rate,
            time_months=time_months,
        )

        expected_tokens = total_airdrop_tokens * expected_allocation_share
        if expected_tokens == 0:
            required_token_price = float('inf')
        else:
            required_token_price = cliff_data['risk_adjusted_required_value'] / expected_tokens
        
        assumed_airdrop_percent = 10.0
        total_supply = total_airdrop_tokens / (assumed_airdrop_percent / 100)
        required_market_cap = required_token_price * total_supply

        return {
            "strategy_name": strategy_name,
            "capital": capital,
            "opportunity_cost_rate": opportunity_cost_rate,
            "expected_allocation_share": expected_allocation_share,
            "required_token_price": required_token_price,
            "required_market_cap": required_market_cap,
            "details": cliff_data,
        }
