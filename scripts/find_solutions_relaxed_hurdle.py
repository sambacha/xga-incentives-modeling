#!/usr/bin/env python3
"""
RELAXED HURDLE RATE SOLUTION FINDER

After mathematical analysis showing hurdle rate < 5.0 is impossible,
this script finds optimal solutions with progressively relaxed hurdle constraints:

1. Hurdle Rate < 6.0 (aggressive but feasible)
2. Hurdle Rate < 7.0 (balanced)  
3. Hurdle Rate < 8.0 (conservative)
4. Minimize hurdle rate (optimization target)
5. Best overall solution (multi-objective)

But wait, maybe it's better... to understand WHY the constraint is impossible
and work backwards from the mathematical limits to find the best achievable solutions.

Mathematical insight: For hurdle rate = β/(β-1) < 5.0, we need β > 1.25
But with realistic opportunity costs (2-10%) and volatilities (30-200%),
the beta values are typically much higher, leading to hurdle rates of 8-15x.
"""
import logging
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import random

from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints, AirdropParameters
from airdrop_calculator.core import AirdropCalculator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RelaxedHurdleSolutionFinder:
    """
    Finds optimal solutions with relaxed hurdle rate constraints.
    
    Explores multiple hurdle rate thresholds to find the best achievable
    balance between economic viability and user profitability.
    """
    
    def __init__(self):
        self.solver = EnhancedZ3Solver()
        self.calculator = AirdropCalculator(self.solver)
        
    def analyze_achievable_hurdle_rates(self) -> Dict[float, List[Tuple]]:
        """
        Analyze what hurdle rates are actually achievable with different parameters.
        
        Returns viable combinations for different hurdle rate thresholds.
        """
        logger.info("🔍 Analyzing achievable hurdle rate boundaries...")
        
        # Test different parameter combinations
        opportunity_costs = np.linspace(2.0, 12.0, 30)
        volatilities = np.linspace(40.0, 250.0, 30)
        
        hurdle_buckets = {
            6.0: [],
            7.0: [],
            8.0: [],
            10.0: [],
            15.0: []
        }
        
        all_hurdles = []
        
        for opp_cost in opportunity_costs:
            for volatility in volatilities:
                try:
                    hurdle_rate = self.calculator.calculate_hurdle_rate(opp_cost, volatility)
                    all_hurdles.append(hurdle_rate)
                    
                    # Categorize by threshold
                    for threshold in sorted(hurdle_buckets.keys()):
                        if hurdle_rate < threshold:
                            hurdle_buckets[threshold].append((opp_cost, volatility, hurdle_rate))
                            break
                            
                except Exception:
                    continue
        
        logger.info("📊 Hurdle Rate Analysis Results:")
        for threshold, combinations in hurdle_buckets.items():
            logger.info(f"   • Hurdle < {threshold}: {len(combinations)} combinations")
        
        if all_hurdles:
            min_hurdle = min(all_hurdles)
            avg_hurdle = np.mean(all_hurdles)
            logger.info(f"   • Minimum achievable hurdle: {min_hurdle:.2f}x")
            logger.info(f"   • Average hurdle rate: {avg_hurdle:.2f}x")
        
        return hurdle_buckets
    
    def approach_1_hurdle_below_6(self, viable_params: List[Tuple]) -> Optional[AirdropParameters]:
        """APPROACH 1: Target hurdle rate < 6.0 (most aggressive)"""
        logger.info("🎯 APPROACH 1: Hurdle Rate < 6.0 (Aggressive)")
        
        if not viable_params:
            logger.warning("No viable parameters for hurdle < 6.0")
            return None
        
        return self._solve_with_hurdle_constraint(viable_params, 6.0, "aggressive")
    
    def approach_2_hurdle_below_7(self, viable_params: List[Tuple]) -> Optional[AirdropParameters]:
        """APPROACH 2: Target hurdle rate < 7.0 (balanced)"""
        logger.info("⚖️ APPROACH 2: Hurdle Rate < 7.0 (Balanced)")
        
        if not viable_params:
            logger.warning("No viable parameters for hurdle < 7.0")
            return None
        
        return self._solve_with_hurdle_constraint(viable_params, 7.0, "balanced")
    
    def approach_3_hurdle_below_8(self, viable_params: List[Tuple]) -> Optional[AirdropParameters]:
        """APPROACH 3: Target hurdle rate < 8.0 (conservative)"""
        logger.info("🛡️ APPROACH 3: Hurdle Rate < 8.0 (Conservative)")
        
        if not viable_params:
            logger.warning("No viable parameters for hurdle < 8.0")
            return None
        
        return self._solve_with_hurdle_constraint(viable_params, 8.0, "conservative")
    
    def approach_4_minimize_hurdle(self, viable_params: List[Tuple]) -> Optional[AirdropParameters]:
        """APPROACH 4: Minimize hurdle rate (optimization target)"""
        logger.info("📉 APPROACH 4: Minimize Hurdle Rate")
        
        if not viable_params:
            logger.warning("No viable parameters for hurdle minimization")
            return None
        
        # Find the parameter combination with lowest hurdle rate
        best_params = min(viable_params, key=lambda x: x[2])
        opp_cost, volatility, expected_hurdle = best_params
        
        logger.info(f"   Using optimal params: OppCost {opp_cost:.2f}%, Vol {volatility:.1f}%")
        
        return self._solve_with_specific_params(opp_cost, volatility, "minimize_hurdle")
    
    def approach_5_best_overall(self, all_viable_params: Dict[float, List[Tuple]]) -> Optional[AirdropParameters]:
        """APPROACH 5: Best overall solution (multi-objective optimization)"""
        logger.info("🏆 APPROACH 5: Best Overall Solution")
        
        # Try different hurdle thresholds and find best overall solution
        best_solution = None
        best_score = 0
        
        for threshold in sorted(all_viable_params.keys()):
            viable_params = all_viable_params[threshold]
            if not viable_params:
                continue
            
            logger.info(f"   Trying hurdle threshold < {threshold} ({len(viable_params)} combinations)")
            
            # Try top 5 parameter combinations for this threshold
            for opp_cost, volatility, hurdle_rate in viable_params[:5]:
                try:
                    solution = self._solve_with_specific_params(opp_cost, volatility, f"threshold_{threshold}")
                    
                    if solution:
                        score = self._comprehensive_score(solution, hurdle_rate)
                        if score > best_score:
                            best_solution = solution
                            best_score = score
                            logger.info(f"     New best: Score {score:.2f}, Hurdle {hurdle_rate:.2f}x")
                            
                except Exception as e:
                    logger.debug(f"     Failed combination: {e}")
                    continue
        
        return best_solution
    
    def _solve_with_hurdle_constraint(self, viable_params: List[Tuple], max_hurdle: float, strategy: str) -> Optional[AirdropParameters]:
        """Solve with specific hurdle rate constraint."""
        
        best_solution = None
        best_score = 0
        
        # Try multiple parameter combinations
        for opp_cost, volatility, hurdle_rate in viable_params[:8]:  # Try top 8
            if hurdle_rate >= max_hurdle:
                continue
            
            try:
                solution = self._solve_with_specific_params(opp_cost, volatility, strategy)
                
                if solution:
                    # Verify hurdle rate is still acceptable
                    actual_hurdle = self.calculator.calculate_hurdle_rate(
                        solution.opportunity_cost, solution.volatility
                    )
                    
                    if actual_hurdle < max_hurdle:
                        score = self._comprehensive_score(solution, actual_hurdle)
                        if score > best_score:
                            best_solution = solution
                            best_score = score
                            logger.info(f"   New best {strategy}: Score {score:.2f}, Hurdle {actual_hurdle:.2f}x")
                            
            except Exception as e:
                logger.debug(f"   Failed: {e}")
                continue
        
        return best_solution
    
    def _solve_with_specific_params(self, opportunity_cost: float, volatility: float, strategy: str) -> Optional[AirdropParameters]:
        """Solve with specific opportunity cost and volatility parameters."""
        
        # Adjust constraints based on strategy
        if strategy == "aggressive":
            constraints = SolverConstraints(
                min_supply=3_000_000_000,
                max_supply=60_000_000_000,
                min_price=0.001,
                max_price=0.45,
                min_airdrop_percent=40.0,
                max_airdrop_percent=75.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility,
                gas_cost=15.0,
                campaign_duration=9
            )
        elif strategy == "balanced":
            constraints = SolverConstraints(
                min_supply=2_500_000_000,
                max_supply=40_000_000_000,
                min_price=0.01,
                max_price=0.40,
                min_airdrop_percent=35.0,
                max_airdrop_percent=65.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility,
                gas_cost=25.0,
                campaign_duration=7
            )
        elif strategy == "conservative":
            constraints = SolverConstraints(
                min_supply=2_000_000_000,
                max_supply=25_000_000_000,
                min_price=0.05,
                max_price=0.35,
                min_airdrop_percent=30.0,
                max_airdrop_percent=55.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility,
                gas_cost=35.0,
                campaign_duration=6
            )
        else:  # minimize_hurdle or threshold strategies
            constraints = SolverConstraints(
                min_supply=2_000_000_000,
                max_supply=50_000_000_000,
                min_price=0.001,
                max_price=0.49,
                min_airdrop_percent=30.0,
                max_airdrop_percent=70.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility,
                gas_cost=20.0,
                campaign_duration=8
            )
        
        # Try soft constraints approach
        objectives = {
            'market_cap': (2_200_000_000, 0.25),
            'profitable_users': (92.0, 0.55),
            'launch_price': (0.25, 0.20)
        }
        
        return self.solver.solve_with_soft_constraints(objectives, constraints)
    
    def _comprehensive_score(self, params: AirdropParameters, hurdle_rate: float) -> float:
        """
        Comprehensive scoring that balances all objectives.
        
        Scoring (100 points total):
        - Price below $0.50: 20 points
        - Supply >= 2B: 15 points
        - High airdrop %: 15 points  
        - Low hurdle rate: 25 points (NEW emphasis)
        - User profitability: 25 points
        """
        score = 0
        
        # Price constraint (20 points)
        if params.launch_price < 0.50:
            score += 20 * (0.50 - params.launch_price) / 0.50
        
        # Supply constraint (15 points)
        if params.total_supply >= 2_000_000_000:
            score += 15
        
        # Airdrop percentage (15 points)
        score += min(15, params.airdrop_percent / 70.0 * 15)
        
        # Hurdle rate (25 points) - Higher weight
        if hurdle_rate <= 15.0:  # Reasonable maximum
            score += 25 * (15.0 - hurdle_rate) / 15.0
        
        # User profitability (25 points)
        profitable_pop = self._calculate_profitable_population(params)
        score += (profitable_pop / 100.0) * 25
        
        return score
    
    def _calculate_profitable_population(self, params: AirdropParameters) -> float:
        """Calculate percentage of population that is profitable."""
        profitable_pop = 0.0
        
        for segment in self.calculator.user_segments:
            avg_capital = (segment.min_capital + segment.max_capital) / 2
            
            cliff_value = self.calculator.calculate_cliff_value(
                capital=avg_capital,
                opportunity_cost=params.opportunity_cost,
                volatility=params.volatility,
                time_months=params.campaign_duration,
                gas_cost=params.gas_cost,
                num_transactions=segment.avg_transactions
            )
            
            min_tokens = cliff_value / params.launch_price
            estimated_allocation = self.calculator.estimate_user_allocation(
                capital=avg_capital,
                allocation_model="quadratic",
                total_supply=params.total_supply,
                airdrop_percent=params.airdrop_percent
            )
            
            if estimated_allocation >= min_tokens:
                profitable_pop += segment.population_percent
        
        return profitable_pop
    
    def analyze_solution(self, params: AirdropParameters, approach_name: str) -> Dict:
        """Comprehensive solution analysis."""
        hurdle_rate = self.calculator.calculate_hurdle_rate(params.opportunity_cost, params.volatility)
        market_cap = params.total_supply * params.launch_price
        profitable_pop = self._calculate_profitable_population(params)
        
        invariants = {
            'price_below_050': params.launch_price < 0.50,
            'supply_min_2b': params.total_supply >= 2_000_000_000,
            'opp_cost_min_2': params.opportunity_cost >= 2.0,
            'airdrop_generous': params.airdrop_percent >= 30.0,
            'hurdle_rate_reasonable': hurdle_rate <= 15.0,
            'majority_profitable': profitable_pop >= 75.0,
            'market_cap_viable': 500_000_000 <= market_cap <= 20_000_000_000
        }
        
        quality_score = self._comprehensive_score(params, hurdle_rate)
        
        return {
            "approach": approach_name,
            "parameters": {k: (v.item() if hasattr(v, 'item') else v) for k, v in asdict(params).items()},
            "hurdle_rate": float(hurdle_rate),
            "invariants": {k: bool(v) for k, v in invariants.items()},
            "all_invariants_met": all(invariants.values()),
            "market_cap": float(market_cap),
            "profitable_population": float(profitable_pop),
            "quality_score": float(quality_score),
            "summary": {
                "price": f"${params.launch_price:.3f}",
                "supply": f"{params.total_supply:,.0f}",
                "airdrop": f"{params.airdrop_percent:.1f}%",
                "hurdle_rate": f"{hurdle_rate:.2f}x",
                "market_cap": f"${market_cap:,.0f}",
                "profitable_users": f"{profitable_pop:.1f}%",
                "quality_score": f"{quality_score:.1f}/100"
            }
        }

def main():
    """Execute relaxed hurdle rate solution finding."""
    logger.info("🚀 Starting Relaxed Hurdle Rate Solution Finder")
    logger.info("🎯 Finding best achievable hurdle rates")
    logger.info("=" * 70)
    
    finder = RelaxedHurdleSolutionFinder()
    
    # Step 1: Analyze achievable hurdle rates
    hurdle_buckets = finder.analyze_achievable_hurdle_rates()
    
    # Step 2: Execute approaches with viable parameters
    approaches = [
        ("hurdle_below_6", lambda: finder.approach_1_hurdle_below_6(hurdle_buckets.get(6.0, []))),
        ("hurdle_below_7", lambda: finder.approach_2_hurdle_below_7(hurdle_buckets.get(7.0, []))),
        ("hurdle_below_8", lambda: finder.approach_3_hurdle_below_8(hurdle_buckets.get(8.0, []))),
        ("minimize_hurdle", lambda: finder.approach_4_minimize_hurdle(hurdle_buckets.get(15.0, []))),
        ("best_overall", lambda: finder.approach_5_best_overall(hurdle_buckets))
    ]
    
    solutions = []
    
    for approach_name, approach_func in approaches:
        logger.info(f"\n--- Executing {approach_name} ---")
        
        try:
            solution = approach_func()
            
            if solution:
                analysis = finder.analyze_solution(solution, approach_name)
                solutions.append(analysis)
                
                logger.info(f"✅ {approach_name}: {analysis['summary']['price']}, "
                          f"Hurdle: {analysis['summary']['hurdle_rate']}, "
                          f"Score: {analysis['summary']['quality_score']}, "
                          f"Profitable: {analysis['summary']['profitable_users']}")
            else:
                logger.warning(f"❌ {approach_name}: No solution found")
                
        except Exception as e:
            logger.error(f"❌ {approach_name}: Error - {e}")
    
    # Save results
    output_file = "relaxed_hurdle_solutions.json"
    with open(output_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Display results
    print("\n" + "=" * 85)
    print("RELAXED HURDLE RATE AIRDROP SOLUTIONS")
    print("=" * 85)
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['approach'].upper().replace('_', ' ')} APPROACH")
        print(f"   Price: {solution['summary']['price']} {'✅' if solution['invariants']['price_below_050'] else '❌'}")
        print(f"   Supply: {solution['summary']['supply']} {'✅' if solution['invariants']['supply_min_2b'] else '❌'}")
        print(f"   Airdrop: {solution['summary']['airdrop']} {'✅' if solution['invariants']['airdrop_generous'] else '❌'}")
        print(f"   Hurdle Rate: {solution['summary']['hurdle_rate']} {'✅' if solution['invariants']['hurdle_rate_reasonable'] else '❌'}")
        print(f"   Market Cap: {solution['summary']['market_cap']}")
        print(f"   Profitable Users: {solution['summary']['profitable_users']}")
        print(f"   Quality Score: {solution['summary']['quality_score']}")
        print(f"   All Constraints: {'✅' if solution['all_invariants_met'] else '❌'}")
    
    # Analysis and recommendations
    print("\n" + "=" * 85)
    print("HURDLE RATE OPTIMIZATION ANALYSIS:")
    
    if solutions:
        # Find best by hurdle rate
        best_hurdle = min(solutions, key=lambda x: x['hurdle_rate'])
        print(f"\n🎯 LOWEST HURDLE RATE: {best_hurdle['hurdle_rate']:.2f}x ({best_hurdle['approach']})")
        
        # Find best overall
        valid_solutions = [s for s in solutions if s['all_invariants_met']]
        if valid_solutions:
            best_overall = max(valid_solutions, key=lambda x: x['quality_score'])
            print(f"\n🏆 BEST OVERALL SOLUTION: {best_overall['approach']} ({best_overall['summary']['quality_score']})")
            print(f"   • Price: {best_overall['summary']['price']}")
            print(f"   • Hurdle Rate: {best_overall['summary']['hurdle_rate']}")
            print(f"   • Profitable Users: {best_overall['summary']['profitable_users']}")
            
        # Hurdle rate distribution
        hurdle_rates = [s['hurdle_rate'] for s in solutions]
        print(f"\n📊 HURDLE RATE DISTRIBUTION:")
        print(f"   • Minimum: {min(hurdle_rates):.2f}x")
        print(f"   • Average: {np.mean(hurdle_rates):.2f}x")
        print(f"   • Maximum: {max(hurdle_rates):.2f}x")
        
        # Success rate by approach
        success_rate = len(valid_solutions) / len(solutions) * 100
        print(f"   • Success rate (all constraints): {success_rate:.1f}%")
        
    return solutions

if __name__ == "__main__":
    solutions = main()