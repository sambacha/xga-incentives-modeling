import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import logging

from .types import AirdropParameters, UserSegment, SegmentResult, MarketMetrics, Scenario, SolverConstraints, TrackType, TrackParameters
from .solver import EnhancedZ3Solver
from .utils import safe_division, safe_multiplication
from .tracks import MultiTrackCalculator, create_track_calculator

logger = logging.getLogger(__name__)

class AirdropCalculator:
    """Main calculator class implementing exotic options framework"""
    
    def __init__(self, params: AirdropParameters):
        self.params = params
        self.risk_free_base = 0.02
        self._beta_cache = {}  # Internal cache for beta calculations
        
        # Define user segments with validation
        self.user_segments = [
            UserSegment("Retail (<$1k)", 100, 1000, 60, 5),
            UserSegment("Power Users ($1-10k)", 1000, 10000, 30, 20),
            UserSegment("Whales ($10-100k)", 10000, 100000, 8, 50),
            UserSegment("Institutions (>$100k)", 100000, 1000000, 2, 100)
        ]
    
    @lru_cache(maxsize=256)
    def calculate_beta(self, r: float, sigma: float, delta: float = 0.0) -> float:
        """
        Calculate beta parameter for perpetual American option pricing
        
        Using the correct formula from option pricing theory:
        β = 0.5 - (r-δ)/σ² + √[(0.5 - (r-δ)/σ²)² + 2r/σ²]
        
        Where:
        - r is the risk-free rate (opportunity cost)
        - σ is the volatility
        - δ is the dividend yield (default 0 for airdrops)
        """
        # Ensure inputs are within reasonable ranges
        r = max(0.001, min(r, 0.5))  # Cap between 0.1% and 50%
        sigma = max(0.1, min(sigma, 2.0))  # Cap between 10% and 200%
        delta = max(0, min(delta, r))  # Delta can't exceed r
        
        sigma_squared = sigma ** 2
        
        # Correct formula for beta in perpetual American options
        a = 0.5 - (r - delta) / sigma_squared
        
        # Calculate discriminant
        discriminant = a * a + 2 * r / sigma_squared
        
        if discriminant < 0:
            logger.warning(f"Negative discriminant in beta calculation: {discriminant}")
            # This should not happen with proper inputs, but return safe default
            return 1.5
        
        # Calculate beta
        beta = a + np.sqrt(discriminant)
        
        # Ensure beta > 1 for meaningful hurdle rates
        # Beta must be strictly greater than 1 for the hurdle rate formula to work
        if beta <= 1.0:
            logger.warning(f"Beta {beta} <= 1.0, adjusting to minimum viable value")
            return 1.2  # Minimum viable beta
        
        # Cap at reasonable maximum for airdrops
        return min(beta, 3.0)
    
    @lru_cache(maxsize=256)
    def calculate_hurdle_rate(self, opportunity_cost: float, volatility: float) -> float:
        """
        Calculate hurdle rate multiple with robust error handling
        
        Hurdle rate = β/(β-1) for perpetual American options
        """
        r = opportunity_cost / 100
        sigma = volatility / 100
        
        try:
            beta = self.calculate_beta(r, sigma)
            
            # Beta must be strictly greater than 1 for hurdle rate to be defined
            if beta <= 1.0:
                logger.warning(f"Beta {beta} <= 1.0, cannot calculate hurdle rate")
                return float('inf')  # Infinite hurdle rate when beta <= 1
            
            # Check if beta is too close to 1 (would give extreme hurdle rates)
            if beta < 1.01:
                logger.warning(f"Beta {beta} very close to 1, returning maximum hurdle rate")
                return 10.0  # Cap at maximum reasonable hurdle rate
            
            hurdle = beta / (beta - 1)
            
            # Bound hurdle rate to realistic values
            # For viable airdrops, hurdle should be between 1.1 and 10.0
            # This represents 10% to 900% required returns
            if hurdle < 1.1:
                logger.warning(f"Calculated hurdle rate {hurdle} < 1.1, adjusting to minimum")
                return 1.1
            elif hurdle > 10.0:
                logger.warning(f"Calculated hurdle rate {hurdle} > 10.0, capping at maximum")
                return 10.0
            
            return hurdle
        except Exception as e:
            logger.error(f"Error calculating hurdle rate: {e}")
            return 2.0  # Reasonable default for airdrops
    
    def calculate_cliff_value(self, capital: float, opportunity_cost: float, 
                            volatility: float, time_months: int, gas_cost: float,
                            num_transactions: int = 10) -> float:
        """
        Calculate minimum value needed for profitability (cliff)
        
        But wait, maybe it's better... to include a safety margin
        for unexpected costs
        """
        r = opportunity_cost / 100
        time_years = time_months / 12
        
        # Opportunity cost
        opp_cost = capital * r * time_years
        
        # Transaction costs with safety margin
        safety_margin = 1.2  # 20% safety margin for gas price volatility
        tx_costs = gas_cost * num_transactions * safety_margin
        
        # Hurdle multiple
        hurdle = self.calculate_hurdle_rate(opportunity_cost, volatility)
        
        # Required value before risk adjustment
        required_value = capital * hurdle + opp_cost + tx_costs
        
        # Risk adjust for probability
        # Use the airdrop certainty from parameters
        certainty_factor = self.params.airdrop_certainty / 100
        if certainty_factor <= 0:
            return float('inf')
        
        risk_adjusted = required_value / certainty_factor
        
        return risk_adjusted
    
    def calculate_min_profitable_tokens(self, capital: float, 
                                      user_segment: Optional[UserSegment] = None) -> float:
        """Calculate minimum tokens needed for profitability"""
        num_tx = user_segment.avg_transactions if user_segment else 10
        
        # Use parameter values for calculation
        cliff_value = self.calculate_cliff_value(
            capital,
            self.params.opportunity_cost,
            self.params.volatility,
            self.params.campaign_duration,
            self.params.gas_cost,
            num_tx
        )
        
        return safe_division(cliff_value, self.params.launch_price, float('inf'))
    
    def estimate_user_allocation(self, capital: float, 
                               allocation_model: str = "quadratic",
                               total_supply: Optional[float] = None,
                               airdrop_percent: Optional[float] = None) -> float:
        """
        Estimate token allocation for a given capital amount
        
        But wait, maybe it's better... to add more sophisticated
        allocation models based on actual airdrop data and allow parameter injection
        """
        # Use parameter values if not provided
        if total_supply is None:
            total_supply = self.params.total_supply
        if airdrop_percent is None:
            airdrop_percent = self.params.airdrop_percent
        
        # Calculate total airdrop tokens
        total_airdrop = total_supply * (airdrop_percent / 100)
        
        if allocation_model == "linear":
            allocation_ratio = capital / 1_000_000
            return total_airdrop * allocation_ratio * 0.01
        
        elif allocation_model == "quadratic":
            # Quadratic funding style with diminishing returns
            sqrt_capital = np.sqrt(min(capital, 1_000_000))
            allocation_ratio = sqrt_capital / 10_000
            return total_airdrop * allocation_ratio * 0.1
        
        elif allocation_model == "logarithmic":
            # Logarithmic scaling for whale resistance
            log_capital = np.log10(max(capital, 100))
            allocation_ratio = log_capital / 6
            return total_airdrop * allocation_ratio * 0.05
        
        elif allocation_model == "tiered":
            # Tiered allocation based on capital ranges
            if capital < 1_000:
                multiplier = 1.5
            elif capital < 10_000:
                multiplier = 1.2
            elif capital < 100_000:
                multiplier = 1.0
            else:
                multiplier = 0.8
            
            base_allocation = (capital / 500_000) * total_airdrop * 0.01
            return base_allocation * multiplier
        
        else:
            raise ValueError(f"Unknown allocation model: {allocation_model}")
    
    def analyze_segment_profitability(self, segment: UserSegment, 
                                    allocation_model: str = "quadratic") -> SegmentResult:
        """Analyze profitability for a user segment"""
        avg_capital = (segment.min_capital + segment.max_capital) / 2
        
        min_tokens = self.calculate_min_profitable_tokens(avg_capital, segment)
        estimated_allocation = self.estimate_user_allocation(avg_capital, allocation_model)
        
        profitable = estimated_allocation >= min_tokens
        
        if profitable:
            gross_value = estimated_allocation * self.params.launch_price
            total_cost = avg_capital + (self.params.gas_cost * segment.avg_transactions)
            # ROI calculation: if cost is zero or negative, ROI is undefined
            if total_cost <= 0:
                roi = float('nan')  # Undefined ROI when no cost
            else:
                roi = ((gross_value - total_cost) / total_cost) * 100
        else:
            roi = -100.0  # Total loss when not profitable
        
        return SegmentResult(
            segment=segment.name,
            avg_capital=avg_capital,
            min_tokens=min_tokens,
            estimated_allocation=estimated_allocation,
            profitable=profitable,
            roi=roi,
            population_percent=segment.population_percent
        )
    
    def run_scenario(self, scenario: Scenario) -> Optional[Dict]:
        """Runs a single scenario and returns the results."""
        try:
            constraints = SolverConstraints(**scenario.constraints)
            
            objectives = {
                k: (v['target'], v['weight'])
                for k, v in scenario.objectives.items()
            }
            
            params = self.solver.solve_with_soft_constraints(objectives, constraints)
            
            if not params:
                logger.warning(f"No solution found for scenario: {scenario.name}")
                return None
            
            impact = self._estimate_impact(params)
            
            return {
                "scenario_name": scenario.name,
                "parameters": params,
                "estimated_impact": impact
            }
        except Exception as e:
            logger.error(f"Error running scenario {scenario.name}: {e}")
            return None

    def compare_scenarios(self, scenarios: List[Scenario]) -> Dict:
        """Compares multiple scenarios and provides a summary."""
        scenario_results = {}
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            if result:
                scenario_results[scenario.name] = result
        
        # Basic comparison logic
        comparison_summary = "Comparison based on market cap:\n"
        sorted_scenarios = sorted(
            scenario_results.items(),
            key=lambda item: item[1]['estimated_impact']['market_cap'],
            reverse=True
        )
        
        for name, result in sorted_scenarios:
            market_cap = result['estimated_impact']['market_cap']
            comparison_summary += f"- {name}: Market Cap = ${market_cap:,.2f}\n"
            
        return {
            "scenario_results": scenario_results,
            "comparison_summary": comparison_summary
        }

    def _estimate_impact(self, params: AirdropParameters) -> Dict:
        """Estimates the market and user impact of a given parameter set."""
        market_cap = safe_multiplication(params.total_supply, params.launch_price)
        
        # Simplified profitability metric
        user_profitability = (30 - params.opportunity_cost) * params.airdrop_percent
        
        # Simplified investment metric
        required_investment = params.hurdle_rate * 1000  # For a $1k user
        
        return {
            "market_cap": market_cap,
            "user_profitability": user_profitability,
            "required_investment": required_investment
        }

    def calculate_market_metrics(self) -> MarketMetrics:
        """Calculate comprehensive market metrics"""
        # Calculate beta for reporting
        # Use parameter values for calculation
        opportunity_cost = self.params.opportunity_cost
        volatility = self.params.volatility
        
        r = opportunity_cost / 100
        sigma = volatility / 100
        beta = self.calculate_beta(r, sigma)
        
        # Hurdle rate
        hurdle_rate = self.calculate_hurdle_rate(opportunity_cost, volatility)
        
        # Typical user analysis
        typical_capital = 5_000
        min_tokens = self.calculate_min_profitable_tokens(typical_capital)
        
        # Minimum market cap calculation
        required_supply = min_tokens / 0.0001  # Assume 0.01% allocation
        min_market_cap = safe_multiplication(required_supply, self.params.launch_price)
        
        # Analyze all segments
        segment_results = []
        total_profitable = 0.0
        weighted_roi = 0.0
        
        for segment in self.user_segments:
            result = self.analyze_segment_profitability(segment)
            segment_results.append(result)
            
            if result.profitable:
                total_profitable += result.population_percent
                weighted_roi += result.roi * (result.population_percent / 100)
        
        # Find optimal capital
        optimal_capital = self.find_optimal_capital()
        
        return MarketMetrics(
            min_market_cap=min_market_cap,
            hurdle_rate=hurdle_rate,
            typical_user_break_even=min_tokens,
            profitable_users_percent=total_profitable,
            avg_roi=weighted_roi,
            optimal_capital=optimal_capital,
            segment_results=segment_results,
            beta_value=beta,
            required_return_multiple=hurdle_rate
        )
    
    def find_optimal_capital(self) -> float:
        """
        Find optimal capital commitment that maximizes ROI
        
        But wait, maybe it's better... to use a more sophisticated
        optimization that considers variance as well as return
        """
        def negative_sharpe_ratio(capital):
            if capital < 100:
                return 1000
            
            min_tokens = self.calculate_min_profitable_tokens(capital)
            estimated_allocation = self.estimate_user_allocation(capital)
            
            if estimated_allocation < min_tokens:
                return 1000
            
            gross_value = estimated_allocation * self.params.launch_price
            total_cost = capital + (self.params.gas_cost * 20)
            roi = (gross_value - total_cost) / total_cost
            
            # Estimate volatility impact on returns
            return_volatility = self.params.volatility / 100 * roi
            
            # Sharpe-like ratio (return / risk)
            if return_volatility > 0:
                sharpe = roi / return_volatility
                return -sharpe
            else:
                return -roi
        
        result = minimize_scalar(
            negative_sharpe_ratio, 
            bounds=(1_000, 100_000), 
            method='bounded',
            options={'xatol': 100}
        )
        
        return result.x
    
    def calculate_track_allocation(self, track_params: TrackParameters, 
                                 airdrop_params: Optional[AirdropParameters] = None) -> Dict:
        """Calculate allocation for a single track"""
        calculator = create_track_calculator(track_params)
        points = calculator.calculate_points()
        capital_equiv = calculator.points_to_capital_equivalent(points)
        risk_factor = calculator.get_risk_factor()
        
        # Use provided params or create default ones
        if airdrop_params is None:
            airdrop_params = AirdropParameters(
                total_supply=1_000_000_000,
                airdrop_percent=10.0,
                launch_price=0.1,
                opportunity_cost=10.0,
                volatility=80.0 * risk_factor,  # Adjust volatility by track risk
                gas_cost=50.0,
                campaign_duration=12,
                airdrop_certainty=90.0,
                revenue_share=0.0,
                vesting_months=18,
                immediate_unlock=30.0
            )
        
        # Calculate allocation using track's preferred model
        allocation_model = calculator.get_allocation_model()
        estimated_allocation = self.estimate_user_allocation(capital_equiv, allocation_model)
        
        # Calculate profitability metrics
        min_tokens = self.calculate_min_profitable_tokens(capital_equiv)
        is_profitable = estimated_allocation >= min_tokens
        
        # Calculate ROI
        gross_value = estimated_allocation * airdrop_params.launch_price
        total_cost = capital_equiv + (airdrop_params.gas_cost * 10)  # Assume 10 transactions
        # ROI calculation: undefined when cost is zero or negative
        if total_cost <= 0:
            roi = float('nan')
        else:
            roi = ((gross_value - total_cost) / total_cost) * 100
        
        return {
            'track_type': track_params.track_type.name,
            'points': points,
            'capital_equivalent': capital_equiv,
            'risk_factor': risk_factor,
            'allocation_model': allocation_model,
            'estimated_allocation': estimated_allocation,
            'min_profitable_tokens': min_tokens,
            'is_profitable': is_profitable,
            'roi': roi,
            'gross_value': gross_value,
            'total_cost': total_cost
        }
    
    def calculate_multi_track_allocation(self, track_params_list: List[TrackParameters],
                                       airdrop_params: Optional[AirdropParameters] = None) -> Dict:
        """Calculate allocation for multiple tracks"""
        if not track_params_list:
            raise ValueError("At least one track parameter must be provided")
        
        # Create multi-track calculator
        multi_calc = MultiTrackCalculator(track_params_list)
        
        # Use provided params or create default ones
        if airdrop_params is None:
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
        
        # Calculate multi-track result
        result = multi_calc.calculate_multi_track_allocation(airdrop_params)
        
        # Calculate additional metrics
        min_tokens = self.calculate_min_profitable_tokens(result.total_capital_equivalent)
        is_profitable = result.recommended_allocation >= min_tokens
        
        # Calculate overall ROI
        gross_value = result.recommended_allocation * airdrop_params.launch_price
        total_cost = result.total_capital_equivalent + (airdrop_params.gas_cost * 20)  # More transactions for multi-track
        # ROI calculation: undefined when cost is zero or negative
        if total_cost <= 0:
            roi = float('nan')
        else:
            roi = ((gross_value - total_cost) / total_cost) * 100
        
        return {
            'track_summary': multi_calc.get_track_summary(),
            'total_points': result.total_points,
            'total_capital_equivalent': result.total_capital_equivalent,
            'weighted_risk_factor': result.weighted_risk_factor,
            'recommended_allocation': result.recommended_allocation,
            'allocation_breakdown': result.allocation_breakdown,
            'min_profitable_tokens': min_tokens,
            'is_profitable': is_profitable,
            'overall_roi': roi,
            'gross_value': gross_value,
            'total_cost': total_cost
        }
    
    def compare_track_strategies(self, scenarios: List[Dict[str, any]]) -> Dict:
        """Compare different track participation strategies"""
        results = []
        
        for scenario in scenarios:
            name = scenario.get('name', 'Unnamed')
            track_params_list = scenario.get('tracks', [])
            airdrop_params = scenario.get('airdrop_params')
            
            try:
                result = self.calculate_multi_track_allocation(track_params_list, airdrop_params)
                results.append({
                    'scenario_name': name,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'scenario_name': name,
                    'success': False,
                    'error': str(e)
                })
        
        # Find best scenario by ROI
        successful_results = [r for r in results if r['success']]
        if successful_results:
            best_scenario = max(successful_results, key=lambda x: x['result']['overall_roi'])
        else:
            best_scenario = None
        
        return {
            'scenarios': results,
            'best_scenario': best_scenario,
            'comparison_summary': self._generate_track_comparison_summary(results)
        }
    
    def _generate_track_comparison_summary(self, results: List[Dict]) -> str:
        """Generate summary of track strategy comparison"""
        summary = "Track Strategy Comparison:\n\n"
        
        for result in results:
            name = result['scenario_name']
            if result['success']:
                data = result['result']
                summary += f"{name}:\n"
                summary += f"  - Total Points: {data['total_points']:,.2f}\n"
                summary += f"  - Capital Equivalent: ${data['total_capital_equivalent']:,.2f}\n"
                summary += f"  - Allocation: {data['recommended_allocation']:,.2f} tokens\n"
                summary += f"  - ROI: {data['overall_roi']:.2f}%\n"
                summary += f"  - Profitable: {'Yes' if data['is_profitable'] else 'No'}\n\n"
            else:
                summary += f"{name}: Failed - {result['error']}\n\n"
        
        return summary
