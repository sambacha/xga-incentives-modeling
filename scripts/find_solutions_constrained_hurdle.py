#!/usr/bin/env python3
"""
CONSTRAINED HURDLE RATE SOLUTION FINDER

Extends the multi-approach finder with an additional constraint:
HURDLE RATE < 5.0 (for better economic viability)

Uses the same 5 approaches but with enhanced constraint handling:
1. Direct Z3 optimization with soft constraints
2. Incremental relaxation with push/pop
3. Pareto frontier exploration
4. Nonlinear constraint solving
5. Brute force parameter space search

All approaches now include hurdle rate validation and optimization.
"""
import logging
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import itertools
import random

from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints, AirdropParameters
from airdrop_calculator.core import AirdropCalculator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConstrainedHurdleSolutionFinder:
    """
    Finds profitable solutions with hurdle rate < 5.0 constraint.
    
    Key enhancement: All approaches now validate and optimize for:
    - Hurdle rate = β/(β-1) < 5.0
    - Where β depends on opportunity cost and volatility
    
    But wait, maybe it's better... to also explore the mathematical
    relationship: lower opportunity cost and higher volatility can help
    achieve lower hurdle rates while maintaining profitability.
    """
    
    def __init__(self):
        self.solver = EnhancedZ3Solver()
        self.calculator = AirdropCalculator(self.solver)
        self.max_hurdle_rate = 5.0  # New constraint
        
    def calculate_hurdle_rate_for_params(self, opportunity_cost: float, volatility: float) -> float:
        """Calculate hurdle rate for given parameters."""
        return self.calculator.calculate_hurdle_rate(opportunity_cost, volatility)
    
    def find_viable_parameter_combinations(self, num_samples: int = 100) -> List[Tuple[float, float]]:
        """
        Find opportunity cost and volatility combinations that yield hurdle rate < 5.
        
        This pre-screening helps all approaches focus on viable parameter spaces.
        """
        viable_combinations = []
        
        # Sample parameter space
        for _ in range(num_samples):
            opportunity_cost = random.uniform(2.0, 8.0)  # 2-8% range
            volatility = random.uniform(30.0, 150.0)    # 30-150% range
            
            hurdle_rate = self.calculate_hurdle_rate_for_params(opportunity_cost, volatility)
            
            if hurdle_rate < self.max_hurdle_rate:
                viable_combinations.append((opportunity_cost, volatility))
        
        logger.info(f"Found {len(viable_combinations)} viable parameter combinations (hurdle < 5)")
        return viable_combinations
    
    def approach_1_soft_constraints_hurdle(self) -> Optional[AirdropParameters]:
        """
        APPROACH 1: Soft Constraints with Hurdle Rate Optimization
        
        Enhanced with hurdle rate as a soft constraint objective.
        """
        logger.info("🎯 APPROACH 1: Soft Constraints + Hurdle Rate Optimization")
        
        # Find viable parameter combinations first
        viable_params = self.find_viable_parameter_combinations(50)
        
        if not viable_params:
            logger.warning("No viable parameter combinations found for approach 1")
            return None
        
        best_solution = None
        best_score = 0
        
        # Try multiple viable parameter combinations
        for opportunity_cost, volatility in viable_params[:10]:  # Try top 10
            constraints = SolverConstraints(
                min_supply=2_000_000_000,
                max_supply=50_000_000_000,
                min_price=0.001,
                max_price=0.49,
                min_airdrop_percent=30.0,
                max_airdrop_percent=70.0,
                opportunity_cost=opportunity_cost,  # Use viable combination
                volatility=volatility,              # Use viable combination
                gas_cost=None,
                campaign_duration=None
            )
            
            # Define objectives with hurdle rate penalty
            objectives = {
                'market_cap': (2_500_000_000, 0.25),
                'profitable_users': (95.0, 0.45),      # High weight on profitability
                'launch_price': (0.25, 0.30)           # Prefer lower prices
            }
            
            solution = self.solver.solve_with_soft_constraints(objectives, constraints)
            
            if solution:
                hurdle_rate = self.calculate_hurdle_rate_for_params(
                    solution.opportunity_cost, solution.volatility
                )
                
                if hurdle_rate < self.max_hurdle_rate:
                    score = self._score_solution(solution, hurdle_rate)
                    if score > best_score:
                        best_solution = solution
                        best_score = score
                        logger.info(f"   New best: Score {score:.2f}, Hurdle {hurdle_rate:.2f}")
        
        return best_solution
    
    def approach_2_incremental_relaxation_hurdle(self) -> Optional[AirdropParameters]:
        """
        APPROACH 2: Incremental Relaxation with Hurdle Rate Validation
        """
        logger.info("📈 APPROACH 2: Incremental Relaxation + Hurdle Validation")
        
        # Find viable parameters
        viable_params = self.find_viable_parameter_combinations(30)
        
        if not viable_params:
            return None
        
        # Use a viable parameter combination
        opportunity_cost, volatility = viable_params[0]
        
        # Define constraint levels with hurdle rate considerations
        constraint_levels = [
            (1, SolverConstraints(  # Strictest
                min_supply=2_000_000_000, max_supply=8_000_000_000,
                min_price=0.05, max_price=0.35,
                min_airdrop_percent=35.0, max_airdrop_percent=45.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility
            )),
            (2, SolverConstraints(  # Moderate
                min_supply=2_000_000_000, max_supply=20_000_000_000,
                min_price=0.01, max_price=0.40,
                min_airdrop_percent=30.0, max_airdrop_percent=55.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility
            )),
            (3, SolverConstraints(  # Relaxed
                min_supply=2_000_000_000, max_supply=40_000_000_000,
                min_price=0.001, max_price=0.49,
                min_airdrop_percent=25.0, max_airdrop_percent=65.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility
            ))
        ]
        
        solution = self.solver.solve_incremental_with_relaxation(
            target_market_cap=3_000_000_000,
            target_profitable_users=92.0,
            constraint_levels=constraint_levels
        )
        
        # Validate hurdle rate
        if solution:
            hurdle_rate = self.calculate_hurdle_rate_for_params(
                solution.opportunity_cost, solution.volatility
            )
            if hurdle_rate >= self.max_hurdle_rate:
                logger.warning(f"Solution rejected: Hurdle rate {hurdle_rate:.2f} >= 5.0")
                return None
        
        return solution
    
    def approach_3_pareto_hurdle_aware(self) -> Optional[AirdropParameters]:
        """
        APPROACH 3: Pareto Optimization with Hurdle Rate Awareness
        """
        logger.info("⚖️ APPROACH 3: Pareto Optimization + Hurdle Awareness")
        
        # Find viable parameters
        viable_params = self.find_viable_parameter_combinations(20)
        
        if not viable_params:
            return None
        
        best_solution = None
        best_quality = 0
        
        # Try several viable parameter combinations
        for opportunity_cost, volatility in viable_params[:5]:
            constraints = SolverConstraints(
                min_supply=3_000_000_000,
                max_supply=25_000_000_000,
                min_price=0.02,
                max_price=0.45,
                min_airdrop_percent=35.0,
                max_airdrop_percent=60.0,
                opportunity_cost=opportunity_cost,
                volatility=volatility,
                gas_cost=30.0,
                campaign_duration=8
            )
            
            try:
                objectives = ['market_cap', 'profitability']
                pareto_solutions = self.solver.find_pareto_optimal_solutions(
                    objectives=objectives,
                    constraints=constraints,
                    num_solutions=6
                )
                
                if pareto_solutions:
                    # Find best solution with valid hurdle rate
                    for sol in pareto_solutions:
                        hurdle_rate = self.calculate_hurdle_rate_for_params(
                            sol['opportunity_cost'], sol['volatility']
                        )
                        
                        if hurdle_rate < self.max_hurdle_rate:
                            params = AirdropParameters(
                                total_supply=sol['total_supply'],
                                airdrop_percent=sol['airdrop_percent'],
                                launch_price=sol['launch_price'],
                                opportunity_cost=sol['opportunity_cost'],
                                volatility=sol['volatility'],
                                gas_cost=sol['gas_cost'],
                                campaign_duration=sol['campaign_duration'],
                                airdrop_certainty=sol['airdrop_certainty'],
                                revenue_share=10.0,
                                vesting_months=18,
                                immediate_unlock=30.0,
                                hurdle_rate=hurdle_rate
                            )
                            
                            quality = self._score_solution(params, hurdle_rate)
                            if quality > best_quality:
                                best_solution = params
                                best_quality = quality
                                
            except Exception as e:
                logger.debug(f"Pareto attempt failed: {e}")
                continue
        
        return best_solution
    
    def approach_4_nonlinear_hurdle_constrained(self) -> Optional[AirdropParameters]:
        """
        APPROACH 4: Nonlinear Solving with Explicit Hurdle Constraints
        """
        logger.info("🧮 APPROACH 4: Nonlinear Solving + Hurdle Constraints")
        
        # Find optimal parameter combinations
        viable_params = self.find_viable_parameter_combinations(15)
        
        if not viable_params:
            return None
        
        # Try the most promising parameter combination
        opportunity_cost, volatility = viable_params[0]
        
        constraints = SolverConstraints(
            min_supply=4_000_000_000,
            max_supply=30_000_000_000,
            min_price=0.08,
            max_price=0.42,
            min_airdrop_percent=40.0,
            max_airdrop_percent=55.0,
            opportunity_cost=opportunity_cost,
            volatility=volatility,
            gas_cost=25.0,
            campaign_duration=7
        )
        
        solution = self.solver.solve_with_nonlinear_constraints(
            target_market_cap=2_800_000_000,
            target_profitable_users=94.0,
            constraints=constraints
        )
        
        # Validate hurdle rate
        if solution:
            hurdle_rate = self.calculate_hurdle_rate_for_params(
                solution.opportunity_cost, solution.volatility
            )
            if hurdle_rate >= self.max_hurdle_rate:
                logger.warning(f"Nonlinear solution rejected: Hurdle rate {hurdle_rate:.2f}")
                return None
        
        return solution
    
    def approach_5_grid_search_hurdle_optimized(self) -> Optional[AirdropParameters]:
        """
        APPROACH 5: Grid Search Optimized for Hurdle Rate
        
        But wait, maybe it's better... to pre-filter the parameter space
        to only include combinations that satisfy the hurdle rate constraint.
        """
        logger.info("🔍 APPROACH 5: Grid Search + Hurdle Rate Optimization")
        
        # Define parameter ranges optimized for lower hurdle rates
        param_ranges = {
            'total_supply': [3_000_000_000, 8_000_000_000, 15_000_000_000, 30_000_000_000],
            'launch_price': [0.05, 0.12, 0.20, 0.30, 0.45],
            'airdrop_percent': [35.0, 45.0, 55.0, 65.0],
            'gas_cost': [15.0, 25.0, 35.0],
            'campaign_duration': [6, 9, 12]
        }
        
        # Pre-filter opportunity cost and volatility combinations
        viable_params = self.find_viable_parameter_combinations(200)
        
        if not viable_params:
            logger.error("No viable parameter combinations for grid search")
            return None
        
        best_solution = None
        best_score = 0
        attempts = 0
        
        # Smart sampling with hurdle rate pre-screening
        for _ in range(75):  # Try 75 combinations
            # Random sample from parameter ranges
            params = {
                param: random.choice(values) 
                for param, values in param_ranges.items()
            }
            
            # Use a random viable parameter combination
            opportunity_cost, volatility = random.choice(viable_params)
            params['opportunity_cost'] = opportunity_cost
            params['volatility'] = volatility
            
            # Add fixed parameters
            params.update({
                'airdrop_certainty': 75.0,
                'revenue_share': 10.0,
                'vesting_months': 18,
                'immediate_unlock': 30.0
            })
            
            try:
                candidate = AirdropParameters(**params)
                hurdle_rate = self.calculate_hurdle_rate_for_params(
                    candidate.opportunity_cost, candidate.volatility
                )
                
                # Only evaluate if hurdle rate is acceptable
                if hurdle_rate < self.max_hurdle_rate:
                    score = self._score_solution(candidate, hurdle_rate)
                    attempts += 1
                    
                    if score > best_score:
                        best_solution = candidate
                        best_score = score
                        logger.info(f"   New best: {score:.2f} (Price: ${candidate.launch_price:.3f}, Hurdle: {hurdle_rate:.2f})")
                        
            except Exception as e:
                logger.debug(f"   Invalid combination: {e}")
                continue
        
        logger.info(f"   Grid search: {attempts} valid attempts, best score: {best_score:.2f}")
        return best_solution
    
    def _score_solution(self, params: AirdropParameters, hurdle_rate: float) -> float:
        """
        Enhanced scoring that includes hurdle rate optimization.
        
        Scoring criteria (100 points total):
        - Price below $0.50: 20 points
        - Supply >= 2B: 15 points  
        - High airdrop %: 15 points
        - User profitability: 30 points
        - Low hurdle rate: 20 points (NEW)
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
        
        # Hurdle rate constraint (20 points) - NEW
        if hurdle_rate < self.max_hurdle_rate:
            score += 20 * (self.max_hurdle_rate - hurdle_rate) / self.max_hurdle_rate
        
        # User profitability analysis (30 points)
        profitable_pop = self._calculate_profitable_population(params)
        score += (profitable_pop / 100.0) * 30
        
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
    
    def check_enhanced_invariants(self, params: AirdropParameters) -> Dict[str, bool]:
        """Enhanced invariant checking including hurdle rate constraint."""
        invariants = {}
        
        # Core requirements
        invariants['price_below_050'] = params.launch_price < 0.50
        invariants['supply_min_2b'] = params.total_supply >= 2_000_000_000
        invariants['opp_cost_min_2'] = params.opportunity_cost >= 2.0
        invariants['airdrop_generous'] = params.airdrop_percent >= 30.0
        
        # NEW: Hurdle rate constraint
        hurdle_rate = self.calculate_hurdle_rate_for_params(
            params.opportunity_cost, params.volatility
        )
        invariants['hurdle_rate_below_5'] = hurdle_rate < 5.0
        
        # Economic viability
        invariants['hurdle_rate_viable'] = 1.1 <= hurdle_rate <= 10.0
        
        # Market viability
        market_cap = params.total_supply * params.launch_price
        invariants['market_cap_reasonable'] = 500_000_000 <= market_cap <= 15_000_000_000
        
        # User profitability
        profitable_pop = self._calculate_profitable_population(params)
        invariants['majority_profitable'] = profitable_pop >= 75.0
        
        return invariants
    
    def analyze_solution(self, params: AirdropParameters, approach_name: str) -> Dict:
        """Comprehensive solution analysis including hurdle rate."""
        invariants = self.check_enhanced_invariants(params)
        market_cap = params.total_supply * params.launch_price
        profitable_pop = self._calculate_profitable_population(params)
        
        # Calculate hurdle rate
        hurdle_rate = self.calculate_hurdle_rate_for_params(
            params.opportunity_cost, params.volatility
        )
        
        # Calculate quality score
        quality_score = self._score_solution(params, hurdle_rate)
        
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
                "profitable_users": f"{profitable_pop:.1f}%"
            }
        }

def main():
    """Execute all approaches with hurdle rate constraints."""
    logger.info("🚀 Starting Constrained Hurdle Rate Solution Finder")
    logger.info("🎯 NEW CONSTRAINT: Hurdle Rate < 5.0")
    logger.info("=" * 70)
    
    finder = ConstrainedHurdleSolutionFinder()
    
    # Define approaches with their methods
    approaches = [
        ("soft_constraints_hurdle", finder.approach_1_soft_constraints_hurdle),
        ("incremental_relaxation_hurdle", finder.approach_2_incremental_relaxation_hurdle), 
        ("pareto_hurdle_aware", finder.approach_3_pareto_hurdle_aware),
        ("nonlinear_hurdle_constrained", finder.approach_4_nonlinear_hurdle_constrained),
        ("grid_search_hurdle_optimized", finder.approach_5_grid_search_hurdle_optimized)
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
                          f"Score: {analysis['quality_score']:.1f}, "
                          f"Profitable: {analysis['summary']['profitable_users']}")
            else:
                logger.warning(f"❌ {approach_name}: No solution found")
                
        except Exception as e:
            logger.error(f"❌ {approach_name}: Error - {e}")
    
    # Save results
    output_file = "constrained_hurdle_solutions.json"
    with open(output_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Display comprehensive results
    print("\n" + "=" * 85)
    print("CONSTRAINED HURDLE RATE AIRDROP SOLUTIONS (Hurdle Rate < 5.0)")
    print("=" * 85)
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['approach'].upper().replace('_', ' ')} APPROACH")
        print(f"   Price: {solution['summary']['price']} {'✅' if solution['invariants']['price_below_050'] else '❌'}")
        print(f"   Supply: {solution['summary']['supply']} {'✅' if solution['invariants']['supply_min_2b'] else '❌'}")
        print(f"   Airdrop: {solution['summary']['airdrop']} {'✅' if solution['invariants']['airdrop_generous'] else '❌'}")
        print(f"   Hurdle Rate: {solution['summary']['hurdle_rate']} {'✅' if solution['invariants']['hurdle_rate_below_5'] else '❌'}")
        print(f"   Market Cap: {solution['summary']['market_cap']}")
        print(f"   Profitable Users: {solution['summary']['profitable_users']}")
        print(f"   Quality Score: {solution['quality_score']:.1f}%")
        print(f"   All Constraints: {'✅' if solution['all_invariants_met'] else '❌'}")
    
    # Analysis and recommendations
    print("\n" + "=" * 85)
    print("HURDLE RATE CONSTRAINT ANALYSIS:")
    
    valid_solutions = [s for s in solutions if s['all_invariants_met']]
    hurdle_compliant = [s for s in solutions if s['invariants']['hurdle_rate_below_5']]
    
    if valid_solutions:
        best = max(valid_solutions, key=lambda x: x['quality_score'])
        print(f"\n🏆 BEST OVERALL: {best['approach']} ({best['quality_score']:.1f}% score)")
        print(f"   • Price: {best['summary']['price']} ✅")
        print(f"   • Hurdle Rate: {best['summary']['hurdle_rate']} ✅")
        print(f"   • Profitable Users: {best['summary']['profitable_users']}")
        
    print(f"\n📊 HURDLE RATE COMPLIANCE:")
    print(f"   • Solutions meeting hurdle constraint: {len(hurdle_compliant)}/{len(solutions)}")
    print(f"   • Solutions meeting ALL constraints: {len(valid_solutions)}/{len(solutions)}")
    
    if hurdle_compliant:
        avg_hurdle = np.mean([s['hurdle_rate'] for s in hurdle_compliant])
        print(f"   • Average hurdle rate (compliant): {avg_hurdle:.2f}x")
        
        best_hurdle = min(hurdle_compliant, key=lambda x: x['hurdle_rate'])
        print(f"   • Best hurdle rate: {best_hurdle['hurdle_rate']:.2f}x ({best_hurdle['approach']})")
    
    return solutions

if __name__ == "__main__":
    solutions = main()