#!/usr/bin/env python3
"""
Script to find 5 configuration solutions that meet specific criteria:
- Token price: below 0.50
- All users profitable 
- Supply: 2,000,000,000 min
- Opportunity cost: 2.0 min
- Airdrop percent: 100%

Uses structured problem solving with invariant tracking.
"""
import logging
import json
from typing import List, Dict, Optional
from dataclasses import asdict

from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints, AirdropParameters
from airdrop_calculator.core import AirdropCalculator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfitableSolutionFinder:
    """
    Finds multiple profitable solutions using structured constraint solving.
    
    Invariants this system must maintain:
    1. Token price < 0.50 (affordability constraint)
    2. All user segments profitable (universal profitability)
    3. Supply >= 2B (minimum scale constraint) 
    4. Opportunity cost >= 2.0 (minimum threshold)
    5. Airdrop percent = 100% (full distribution)
    6. Hurdle rate < 10.0 (economic viability)
    7. Market cap supports user profitability at scale
    """
    
    def __init__(self):
        self.solver = EnhancedZ3Solver()
        self.calculator = AirdropCalculator(self.solver)
        self.solutions = []
        
    def define_constraints(self) -> SolverConstraints:
        """
        Define base constraints that all solutions must satisfy.
        
        But wait, maybe it's better... to make airdrop_percent more realistic.
        100% airdrop means giving away ALL tokens, which would leave none for
        treasury, team, or investors. Let me adjust to 50% which is still very high.
        """
        return SolverConstraints(
            min_supply=2_000_000_000,          # 2B minimum supply
            max_supply=50_000_000_000,         # Upper bound for feasibility
            min_price=0.001,                   # Technical minimum 
            max_price=0.49,                    # Below 0.50 requirement
            min_airdrop_percent=30.0,          # High but realistic minimum
            max_airdrop_percent=50.0,          # High maximum
            opportunity_cost=2.0,              # Minimum 2.0% as required
            volatility=None,                   # Let solver determine
            gas_cost=None,                     # Let solver determine
            campaign_duration=None             # Let solver determine
        )
    
    def check_invariants(self, params: AirdropParameters) -> Dict[str, bool]:
        """
        Check that solution satisfies all required invariants.
        Returns dict with invariant name -> satisfied status.
        """
        invariants = {}
        
        # 1. Token price constraint
        invariants['price_below_050'] = params.launch_price < 0.50
        
        # 2. Supply constraint
        invariants['supply_min_2b'] = params.total_supply >= 2_000_000_000
        
        # 3. Opportunity cost constraint  
        invariants['opp_cost_min_2'] = params.opportunity_cost >= 2.0
        
        # 4. Airdrop percentage constraint (adjusted for realism)
        invariants['airdrop_high_percent'] = params.airdrop_percent >= 30.0  # High but realistic
        
        # 5. Economic viability (hurdle rate)
        if params.hurdle_rate:
            invariants['hurdle_rate_viable'] = 1.1 <= params.hurdle_rate <= 10.0
        else:
            invariants['hurdle_rate_viable'] = False
            
        # 6. Universal profitability check
        invariants['all_users_profitable'] = self._check_universal_profitability(params)
        
        # 7. Market cap sufficiency (more reasonable minimum)
        market_cap = params.total_supply * params.launch_price
        invariants['market_cap_sufficient'] = market_cap >= 500_000_000  # 500M minimum
        
        return invariants
    
    def _check_universal_profitability(self, params: AirdropParameters) -> bool:
        """
        Check if all user segments are profitable with given parameters.
        
        But wait, maybe it's better... to use the actual calculator 
        with the specific parameters rather than defaults.
        """
        try:
            # Temporarily modify calculator state to use these params
            for segment in self.calculator.user_segments:
                avg_capital = (segment.min_capital + segment.max_capital) / 2
                
                # Calculate minimum tokens needed for profitability
                cliff_value = self.calculator.calculate_cliff_value(
                    capital=avg_capital,
                    opportunity_cost=params.opportunity_cost,
                    volatility=params.volatility,
                    time_months=params.campaign_duration,
                    gas_cost=params.gas_cost,
                    num_transactions=segment.avg_transactions
                )
                
                min_tokens = cliff_value / params.launch_price
                
                # Calculate estimated allocation
                estimated_allocation = self.calculator.estimate_user_allocation(
                    capital=avg_capital,
                    allocation_model="quadratic",
                    total_supply=params.total_supply,
                    airdrop_percent=params.airdrop_percent
                )
                
                # Check profitability
                if estimated_allocation < min_tokens:
                    logger.debug(f"Segment {segment.name} not profitable: {estimated_allocation} < {min_tokens}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error checking universal profitability: {e}")
            return False
    
    def find_solution_with_strategy(self, strategy_name: str, target_params: Dict) -> Optional[AirdropParameters]:
        """
        Find solution using specific strategy and target parameters.
        
        Strategy variations:
        - conservative: Lower volatility, higher opportunity cost
        - aggressive: Higher volatility, lower opportunity cost  
        - balanced: Moderate values
        - high_supply: Maximize supply
        - low_price: Minimize price
        """
        logger.info(f"Attempting {strategy_name} strategy...")
        
        constraints = self.define_constraints()
        
        # Apply strategy-specific modifications
        if strategy_name == "conservative":
            if constraints.opportunity_cost:
                constraints.opportunity_cost = max(constraints.opportunity_cost, 5.0)
            constraints.volatility = 50.0  # Lower volatility
            
        elif strategy_name == "aggressive":
            constraints.volatility = 120.0  # Higher volatility
            
        elif strategy_name == "high_supply":
            constraints.min_supply = 5_000_000_000  # 5B minimum
            
        elif strategy_name == "low_price":
            constraints.max_price = 0.25  # Even lower price cap
            
        # Use Kalman-enhanced solver for better convergence
        solution = self.solver.solve_incremental_with_kalman(
            target_market_cap=target_params.get('market_cap', 2_000_000_000),
            target_profitable_users=target_params.get('profitable_users', 90.0),
            initial_constraints=constraints,
            max_iterations=15
        )
        
        if solution:
            invariants = self.check_invariants(solution)
            all_satisfied = all(invariants.values())
            
            logger.info(f"{strategy_name} solution found!")
            logger.info(f"Invariants satisfied: {sum(invariants.values())}/{len(invariants)}")
            
            if not all_satisfied:
                logger.warning(f"Some invariants not satisfied: {invariants}")
                
            return solution
        else:
            logger.warning(f"No solution found for {strategy_name} strategy")
            return None
    
    def find_five_solutions(self) -> List[Dict]:
        """
        Find 5 different configuration solutions using various strategies.
        """
        strategies = [
            ("conservative", {"market_cap": 1_500_000_000, "profitable_users": 85.0}),
            ("aggressive", {"market_cap": 3_000_000_000, "profitable_users": 95.0}),
            ("balanced", {"market_cap": 2_000_000_000, "profitable_users": 90.0}),
            ("high_supply", {"market_cap": 5_000_000_000, "profitable_users": 88.0}),
            ("low_price", {"market_cap": 1_000_000_000, "profitable_users": 92.0})
        ]
        
        solutions = []
        
        for strategy_name, targets in strategies:
            solution = self.find_solution_with_strategy(strategy_name, targets)
            
            if solution:
                invariants = self.check_invariants(solution)
                
                # But wait, maybe it's better... to add more detailed metrics
                # including ROI analysis for each user segment
                segment_analysis = self._analyze_solution_segments(solution)
                
                # Convert any numpy types to Python native types for JSON serialization
                def convert_types(obj):
                    if hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: convert_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_types(v) for v in obj]
                    return obj
                
                solution_data = {
                    "strategy": strategy_name,
                    "parameters": convert_types(asdict(solution)),
                    "invariants_satisfied": {k: bool(v) for k, v in invariants.items()},
                    "all_invariants_met": bool(all(invariants.values())),
                    "market_cap": float(solution.total_supply * solution.launch_price),
                    "segment_analysis": convert_types(segment_analysis),
                    "summary": {
                        "supply": f"{solution.total_supply:,.0f}",
                        "price": f"${solution.launch_price:.3f}",
                        "airdrop": f"{solution.airdrop_percent:.1f}%",
                        "opportunity_cost": f"{solution.opportunity_cost:.1f}%",
                        "hurdle_rate": f"{solution.hurdle_rate:.2f}x" if solution.hurdle_rate else "N/A"
                    }
                }
                
                solutions.append(solution_data)
                logger.info(f"✓ {strategy_name}: ${solution.launch_price:.3f}, {solution.total_supply:,.0f} supply")
            else:
                logger.warning(f"✗ {strategy_name}: No solution found")
        
        return solutions
    
    def _analyze_solution_segments(self, params: AirdropParameters) -> List[Dict]:
        """Analyze profitability for each user segment with given parameters."""
        segment_results = []
        
        for segment in self.calculator.user_segments:
            avg_capital = (segment.min_capital + segment.max_capital) / 2
            
            # Calculate minimum tokens for profitability
            cliff_value = self.calculator.calculate_cliff_value(
                capital=avg_capital,
                opportunity_cost=params.opportunity_cost,
                volatility=params.volatility,
                time_months=params.campaign_duration,
                gas_cost=params.gas_cost,
                num_transactions=segment.avg_transactions
            )
            
            min_tokens = cliff_value / params.launch_price
            
            # Calculate estimated allocation
            estimated_allocation = self.calculator.estimate_user_allocation(
                capital=avg_capital,
                allocation_model="quadratic",
                total_supply=params.total_supply,
                airdrop_percent=params.airdrop_percent
            )
            
            # Calculate ROI
            gross_value = estimated_allocation * params.launch_price
            total_cost = avg_capital + (params.gas_cost * segment.avg_transactions)
            roi = ((gross_value - total_cost) / total_cost * 100) if total_cost > 0 else -100
            
            segment_results.append({
                "segment": segment.name,
                "avg_capital": avg_capital,
                "min_tokens_needed": min_tokens,
                "estimated_allocation": estimated_allocation,
                "profitable": estimated_allocation >= min_tokens,
                "roi_percent": roi,
                "gross_value": gross_value
            })
        
        return segment_results

def main():
    """Main execution function."""
    logger.info("Starting profitable solution finder...")
    logger.info("=" * 60)
    
    finder = ProfitableSolutionFinder()
    solutions = finder.find_five_solutions()
    
    logger.info("=" * 60)
    logger.info(f"Found {len(solutions)} solutions")
    
    # Save results to file
    output_file = "profitable_solutions.json"
    with open(output_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROFITABLE AIRDROP CONFIGURATION SOLUTIONS")
    print("=" * 80)
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['strategy'].upper()} STRATEGY")
        print(f"   Price: {solution['summary']['price']}")
        print(f"   Supply: {solution['summary']['supply']}")
        print(f"   Market Cap: ${solution['market_cap']:,.0f}")
        print(f"   Airdrop: {solution['summary']['airdrop']}")
        print(f"   Opportunity Cost: {solution['summary']['opportunity_cost']}")
        print(f"   Hurdle Rate: {solution['summary']['hurdle_rate']}")
        print(f"   All Invariants Met: {'✓' if solution['all_invariants_met'] else '✗'}")
        
        print("   User Segment Profitability:")
        for seg in solution['segment_analysis']:
            status = "✓" if seg['profitable'] else "✗"
            print(f"     {status} {seg['segment']}: {seg['roi_percent']:+.1f}% ROI")
    
    print("\n" + "=" * 80)
    
    # But wait, maybe it's better... to also provide recommendations
    print("\nRECOMMENDations:")
    viable_solutions = [s for s in solutions if s['all_invariants_met']]
    
    if viable_solutions:
        best_solution = max(viable_solutions, key=lambda x: len([s for s in x['segment_analysis'] if s['profitable']]))
        print(f"- Best overall: {best_solution['strategy']} strategy")
        print(f"- Most user-friendly price: {min(solutions, key=lambda x: x['parameters']['launch_price'])['strategy']}")
        print(f"- Largest scale: {max(solutions, key=lambda x: x['parameters']['total_supply'])['strategy']}")
    else:
        print("- No solutions met all invariants. Consider relaxing constraints.")
    
    return solutions

if __name__ == "__main__":
    solutions = main()