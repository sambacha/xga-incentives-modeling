#!/usr/bin/env python3
"""
MULTI-APPROACH SOLUTION FINDER

Uses various solver approaches instead of Kalman filter:
1. Direct Z3 optimization with soft constraints
2. Incremental relaxation with push/pop
3. Pareto frontier exploration
4. Nonlinear constraint solving
5. Brute force parameter space search

Each approach targets the same criteria but uses different mathematical strategies.
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

class MultiApproachSolutionFinder:
    """
    Finds profitable solutions using multiple different approaches.
    
    Approaches:
    1. Soft Constraints Optimization - Uses penalty functions
    2. Incremental Relaxation - Systematic constraint loosening
    3. Pareto Optimization - Multi-objective optimization
    4. Nonlinear Solving - Advanced constraint handling
    5. Grid Search - Brute force parameter exploration
    """
    
    def __init__(self):
        self.solver = EnhancedZ3Solver()
        self.calculator = AirdropCalculator(self.solver)
        
    def approach_1_soft_constraints(self) -> Optional[AirdropParameters]:
        """
        APPROACH 1: Soft Constraints Optimization
        
        Uses weighted penalty functions to balance competing objectives.
        But wait, maybe it's better... to use different weights for different objectives
        to prioritize user profitability over strict constraint adherence.
        """
        logger.info("🎯 APPROACH 1: Soft Constraints Optimization")
        
        constraints = SolverConstraints(
            min_supply=2_000_000_000,
            max_supply=50_000_000_000,
            min_price=0.001,
            max_price=0.49,
            min_airdrop_percent=30.0,
            max_airdrop_percent=70.0,
            opportunity_cost=2.0,
            volatility=None,
            gas_cost=None,
            campaign_duration=None
        )
        
        # Define objectives with weights (target_value, weight)
        objectives = {
            'market_cap': (2_000_000_000, 0.3),      # Target 2B market cap, weight 0.3
            'profitable_users': (95.0, 0.5),         # High weight on user profitability
            'launch_price': (0.30, 0.2)              # Prefer lower prices
        }
        
        solution = self.solver.solve_with_soft_constraints(objectives, constraints)
        return solution
    
    def approach_2_incremental_relaxation(self) -> Optional[AirdropParameters]:
        """
        APPROACH 2: Incremental Relaxation with Push/Pop
        
        Starts with strict constraints and systematically relaxes them.
        """
        logger.info("📈 APPROACH 2: Incremental Relaxation")
        
        # Define constraint levels from strict to relaxed
        constraint_levels = [
            (1, SolverConstraints(  # Strictest
                min_supply=2_000_000_000, max_supply=10_000_000_000,
                min_price=0.01, max_price=0.30,
                min_airdrop_percent=40.0, max_airdrop_percent=50.0,
                opportunity_cost=2.0
            )),
            (2, SolverConstraints(  # Moderate
                min_supply=2_000_000_000, max_supply=25_000_000_000,
                min_price=0.001, max_price=0.40,
                min_airdrop_percent=35.0, max_airdrop_percent=60.0,
                opportunity_cost=2.0
            )),
            (3, SolverConstraints(  # Relaxed
                min_supply=2_000_000_000, max_supply=50_000_000_000,
                min_price=0.001, max_price=0.49,
                min_airdrop_percent=30.0, max_airdrop_percent=70.0,
                opportunity_cost=2.0
            ))
        ]
        
        solution = self.solver.solve_incremental_with_relaxation(
            target_market_cap=2_500_000_000,
            target_profitable_users=90.0,
            constraint_levels=constraint_levels
        )
        
        return solution
    
    def approach_3_pareto_optimization(self) -> Optional[AirdropParameters]:
        """
        APPROACH 3: Pareto Frontier Exploration
        
        Finds Pareto-optimal solutions between competing objectives.
        """
        logger.info("⚖️ APPROACH 3: Pareto Optimization")
        
        constraints = SolverConstraints(
            min_supply=3_000_000_000,
            max_supply=30_000_000_000,
            min_price=0.01,
            max_price=0.45,
            min_airdrop_percent=35.0,
            max_airdrop_percent=65.0,
            opportunity_cost=3.0,
            volatility=80.0,
            gas_cost=30.0,
            campaign_duration=8
        )
        
        # Find Pareto solutions between market cap and profitability
        objectives = ['market_cap', 'profitability']
        pareto_solutions = self.solver.find_pareto_optimal_solutions(
            objectives=objectives,
            constraints=constraints,
            num_solutions=8
        )
        
        if pareto_solutions:
            # Select solution with best profitability score
            best_solution = max(pareto_solutions, key=lambda x: x.get('profitability_value', 0))
            
            # Convert to AirdropParameters
            return AirdropParameters(
                total_supply=best_solution['total_supply'],
                airdrop_percent=best_solution['airdrop_percent'],
                launch_price=best_solution['launch_price'],
                opportunity_cost=best_solution['opportunity_cost'],
                volatility=best_solution['volatility'],
                gas_cost=best_solution['gas_cost'],
                campaign_duration=best_solution['campaign_duration'],
                airdrop_certainty=best_solution['airdrop_certainty'],
                revenue_share=10.0,
                vesting_months=18,
                immediate_unlock=30.0
            )
        
        return None
    
    def approach_4_nonlinear_solving(self) -> Optional[AirdropParameters]:
        """
        APPROACH 4: Nonlinear Constraint Solving
        
        Uses advanced Z3 tactics for complex constraint handling.
        """
        logger.info("🧮 APPROACH 4: Nonlinear Constraint Solving")
        
        constraints = SolverConstraints(
            min_supply=5_000_000_000,
            max_supply=40_000_000_000,
            min_price=0.05,
            max_price=0.40,
            min_airdrop_percent=40.0,
            max_airdrop_percent=60.0,
            opportunity_cost=2.5,
            volatility=70.0,
            gas_cost=25.0,
            campaign_duration=6
        )
        
        solution = self.solver.solve_with_nonlinear_constraints(
            target_market_cap=3_000_000_000,
            target_profitable_users=92.0,
            constraints=constraints
        )
        
        return solution
    
    def approach_5_grid_search(self) -> Optional[AirdropParameters]:
        """
        APPROACH 5: Grid Search Optimization
        
        Systematically explores parameter space to find optimal combinations.
        But wait, maybe it's better... to use smart sampling rather than 
        exhaustive grid search for better performance.
        """
        logger.info("🔍 APPROACH 5: Smart Grid Search")
        
        # Define parameter ranges
        param_ranges = {
            'total_supply': [3_000_000_000, 8_000_000_000, 15_000_000_000, 25_000_000_000],
            'launch_price': [0.08, 0.15, 0.25, 0.35, 0.45],
            'airdrop_percent': [35.0, 45.0, 55.0, 65.0],
            'opportunity_cost': [2.0, 3.0, 4.0, 5.0],
            'volatility': [60.0, 80.0, 100.0, 120.0],
            'gas_cost': [15.0, 25.0, 35.0, 50.0],
            'campaign_duration': [6, 9, 12, 15]
        }
        
        best_solution = None
        best_score = 0
        
        # Smart sampling: try 50 random combinations instead of exhaustive search
        for i in range(50):
            # Random sample from each parameter range
            params = {
                param: random.choice(values) 
                for param, values in param_ranges.items()
            }
            
            # Add fixed parameters
            params.update({
                'airdrop_certainty': 75.0,
                'revenue_share': 10.0,
                'vesting_months': 18,
                'immediate_unlock': 30.0
            })
            
            try:
                candidate = AirdropParameters(**params)
                score = self._evaluate_candidate(candidate)
                
                if score > best_score:
                    best_solution = candidate
                    best_score = score
                    logger.info(f"   New best: {score:.2f} (Price: ${candidate.launch_price:.3f})")
                    
            except Exception as e:
                logger.debug(f"   Invalid combination: {e}")
                continue
        
        logger.info(f"   Grid search completed. Best score: {best_score:.2f}")
        return best_solution
    
    def _evaluate_candidate(self, params: AirdropParameters) -> float:
        """
        Evaluate candidate solution quality.
        
        Scoring criteria:
        - Price below $0.50: 25 points
        - Supply >= 2B: 15 points  
        - High airdrop %: up to 20 points
        - User profitability: up to 40 points
        """
        score = 0
        
        # Price constraint (25 points max)
        if params.launch_price < 0.50:
            score += 25 * (0.50 - params.launch_price) / 0.50
        
        # Supply constraint (15 points)
        if params.total_supply >= 2_000_000_000:
            score += 15
        
        # Airdrop percentage (20 points max)
        score += min(20, params.airdrop_percent / 70.0 * 20)
        
        # User profitability analysis (40 points max)
        profitable_segments = 0
        total_population = 0
        
        for segment in self.calculator.user_segments:
            avg_capital = (segment.min_capital + segment.max_capital) / 2
            
            # Calculate if profitable
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
                profitable_segments += segment.population_percent
            
            total_population += segment.population_percent
        
        profitability_score = (profitable_segments / total_population) * 40
        score += profitability_score
        
        return score
    
    def check_solution_invariants(self, params: AirdropParameters) -> Dict[str, bool]:
        """Check invariants for any solution approach."""
        invariants = {}
        
        # Core requirements
        invariants['price_below_050'] = params.launch_price < 0.50
        invariants['supply_min_2b'] = params.total_supply >= 2_000_000_000
        invariants['opp_cost_min_2'] = params.opportunity_cost >= 2.0
        invariants['airdrop_generous'] = params.airdrop_percent >= 30.0
        
        # Economic viability
        if hasattr(params, 'hurdle_rate') and params.hurdle_rate:
            invariants['hurdle_rate_viable'] = 1.1 <= params.hurdle_rate <= 10.0
        else:
            # Calculate hurdle rate
            hurdle = self.calculator.calculate_hurdle_rate(params.opportunity_cost, params.volatility)
            invariants['hurdle_rate_viable'] = 1.1 <= hurdle <= 10.0
        
        # Market viability
        market_cap = params.total_supply * params.launch_price
        invariants['market_cap_reasonable'] = 500_000_000 <= market_cap <= 15_000_000_000
        
        # User profitability
        profitable_pop = self._calculate_profitable_population(params)
        invariants['majority_profitable'] = profitable_pop >= 75.0
        
        return invariants
    
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
        invariants = self.check_solution_invariants(params)
        market_cap = params.total_supply * params.launch_price
        profitable_pop = self._calculate_profitable_population(params)
        
        # Calculate quality score
        quality_score = sum(invariants.values()) / len(invariants) * 100
        
        return {
            "approach": approach_name,
            "parameters": {k: (v.item() if hasattr(v, 'item') else v) for k, v in asdict(params).items()},
            "invariants": {k: bool(v) for k, v in invariants.items()},
            "all_invariants_met": all(invariants.values()),
            "market_cap": float(market_cap),
            "profitable_population": float(profitable_pop),
            "quality_score": float(quality_score),
            "summary": {
                "price": f"${params.launch_price:.3f}",
                "supply": f"{params.total_supply:,.0f}",
                "airdrop": f"{params.airdrop_percent:.1f}%",
                "market_cap": f"${market_cap:,.0f}",
                "profitable_users": f"{profitable_pop:.1f}%"
            }
        }

def main():
    """Execute all approaches and compare results."""
    logger.info("🚀 Starting Multi-Approach Solution Finder")
    logger.info("=" * 70)
    
    finder = MultiApproachSolutionFinder()
    
    # Define approaches with their methods
    approaches = [
        ("soft_constraints", finder.approach_1_soft_constraints),
        ("incremental_relaxation", finder.approach_2_incremental_relaxation), 
        ("pareto_optimization", finder.approach_3_pareto_optimization),
        ("nonlinear_solving", finder.approach_4_nonlinear_solving),
        ("grid_search", finder.approach_5_grid_search)
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
                          f"Quality: {analysis['quality_score']:.1f}%, "
                          f"Profitable: {analysis['summary']['profitable_users']}")
            else:
                logger.warning(f"❌ {approach_name}: No solution found")
                
        except Exception as e:
            logger.error(f"❌ {approach_name}: Error - {e}")
    
    # Save results
    output_file = "multi_approach_solutions.json"
    with open(output_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Display comprehensive results
    print("\n" + "=" * 80)
    print("MULTI-APPROACH AIRDROP SOLUTIONS")
    print("=" * 80)
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['approach'].upper().replace('_', ' ')} APPROACH")
        print(f"   Price: {solution['summary']['price']} {'✅' if solution['invariants']['price_below_050'] else '❌'}")
        print(f"   Supply: {solution['summary']['supply']} {'✅' if solution['invariants']['supply_min_2b'] else '❌'}")
        print(f"   Airdrop: {solution['summary']['airdrop']} {'✅' if solution['invariants']['airdrop_generous'] else '❌'}")
        print(f"   Market Cap: {solution['summary']['market_cap']}")
        print(f"   Profitable Users: {solution['summary']['profitable_users']}")
        print(f"   Quality Score: {solution['quality_score']:.1f}%")
        print(f"   All Invariants: {'✅' if solution['all_invariants_met'] else '❌'}")
    
    # Analysis and recommendations
    print("\n" + "=" * 80)
    print("APPROACH COMPARISON & RECOMMENDATIONS:")
    
    valid_solutions = [s for s in solutions if s['all_invariants_met']]
    
    if valid_solutions:
        best = max(valid_solutions, key=lambda x: x['quality_score'])
        print(f"\n🏆 BEST APPROACH: {best['approach']} ({best['quality_score']:.1f}% quality)")
        print(f"   • Price: {best['summary']['price']} (below $0.50 ✅)")
        print(f"   • Supply: {best['summary']['supply']} (≥2B ✅)")
        print(f"   • Profitable Users: {best['summary']['profitable_users']}")
        
        print(f"\n📊 APPROACH EFFECTIVENESS:")
        for approach in approaches:
            approach_solutions = [s for s in solutions if s['approach'] == approach[0]]
            if approach_solutions:
                score = approach_solutions[0]['quality_score']
                meets_all = approach_solutions[0]['all_invariants_met']
                print(f"   • {approach[0]}: {score:.1f}% {'✅' if meets_all else '❌'}")
            else:
                print(f"   • {approach[0]}: Failed ❌")
        
    else:
        print("\n⚠️  No approach met all invariants perfectly.")
        print("Consider adjusting constraints or exploring hybrid approaches.")
        
        if solutions:
            best_partial = max(solutions, key=lambda x: x['quality_score'])
            print(f"\nBest partial solution: {best_partial['approach']} ({best_partial['quality_score']:.1f}%)")
    
    return solutions

if __name__ == "__main__":
    solutions = main()