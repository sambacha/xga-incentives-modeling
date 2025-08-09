#!/usr/bin/env python3
"""
HURDLE RATE CONSTRAINT ANALYSIS

Analyzes why hurdle rate < 5.0 is difficult to achieve and finds
optimal parameter regions that can satisfy this constraint.

Mathematical insight: Hurdle rate = β/(β-1) where β = a + √(a² + 2r/σ²)
and a = 0.5 - (r-δ)/σ²

For lower hurdle rates, we need:
- Lower opportunity cost (r)
- Higher volatility (σ) 
- But this creates tension with profitability requirements

But wait, maybe it's better... to systematically explore the mathematical
boundaries and find the exact parameter combinations that work.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver

def analyze_hurdle_rate_space():
    """Systematically analyze the hurdle rate parameter space."""
    
    calculator = AirdropCalculator(EnhancedZ3Solver())
    
    # Define parameter ranges
    opportunity_costs = np.linspace(2.0, 10.0, 50)  # 2-10%
    volatilities = np.linspace(30.0, 200.0, 50)     # 30-200%
    
    results = []
    viable_combinations = []
    
    print("🔍 Analyzing hurdle rate parameter space...")
    print("=" * 60)
    
    for i, opp_cost in enumerate(opportunity_costs):
        for j, volatility in enumerate(volatilities):
            hurdle_rate = calculator.calculate_hurdle_rate(opp_cost, volatility)
            
            results.append({
                'opportunity_cost': opp_cost,
                'volatility': volatility,
                'hurdle_rate': hurdle_rate
            })
            
            if hurdle_rate < 5.0:
                viable_combinations.append((opp_cost, volatility, hurdle_rate))
    
    print(f"📊 Analysis Results:")
    print(f"   • Total combinations tested: {len(results):,}")
    print(f"   • Viable combinations (hurdle < 5): {len(viable_combinations)}")
    print(f"   • Viability rate: {len(viable_combinations)/len(results)*100:.2f}%")
    
    if viable_combinations:
        print(f"\n✅ VIABLE PARAMETER COMBINATIONS:")
        print(f"{'Opp Cost':<10} {'Volatility':<12} {'Hurdle Rate':<12}")
        print("-" * 40)
        
        # Sort by hurdle rate
        viable_combinations.sort(key=lambda x: x[2])
        
        for opp_cost, vol, hurdle in viable_combinations[:10]:  # Show top 10
            print(f"{opp_cost:<10.2f} {vol:<12.1f} {hurdle:<12.3f}")
        
        # Find optimal regions
        min_hurdle = min(viable_combinations, key=lambda x: x[2])
        print(f"\n🎯 OPTIMAL COMBINATION:")
        print(f"   • Opportunity Cost: {min_hurdle[0]:.2f}%")
        print(f"   • Volatility: {min_hurdle[1]:.1f}%") 
        print(f"   • Hurdle Rate: {min_hurdle[2]:.3f}x")
        
        return viable_combinations
    else:
        print("❌ No viable combinations found in tested range")
        
        # Find closest to 5.0
        closest = min(results, key=lambda x: abs(x['hurdle_rate'] - 5.0))
        print(f"\n🔍 CLOSEST TO 5.0:")
        print(f"   • Opportunity Cost: {closest['opportunity_cost']:.2f}%")
        print(f"   • Volatility: {closest['volatility']:.1f}%")
        print(f"   • Hurdle Rate: {closest['hurdle_rate']:.3f}x")
        
        return []

def find_extended_viable_combinations():
    """Extended search with broader parameter ranges."""
    
    calculator = AirdropCalculator(EnhancedZ3Solver())
    
    print("\n🔍 Extended parameter space search...")
    print("=" * 60)
    
    viable_combinations = []
    
    # Broader ranges with focus on extremes
    opportunity_costs = np.concatenate([
        np.linspace(2.0, 3.0, 20),    # Low opportunity costs
        np.linspace(3.0, 8.0, 30),    # Medium range
        np.linspace(8.0, 15.0, 20)    # High opportunity costs
    ])
    
    volatilities = np.concatenate([
        np.linspace(50.0, 100.0, 30),  # Medium volatilities
        np.linspace(100.0, 200.0, 40), # High volatilities  
        np.linspace(200.0, 300.0, 30)  # Very high volatilities
    ])
    
    for opp_cost in opportunity_costs:
        for volatility in volatilities:
            try:
                hurdle_rate = calculator.calculate_hurdle_rate(opp_cost, volatility)
                
                if hurdle_rate < 5.0 and hurdle_rate > 1.1:  # Valid range
                    viable_combinations.append((opp_cost, volatility, hurdle_rate))
                    
            except Exception as e:
                continue
    
    print(f"📊 Extended Analysis Results:")
    print(f"   • Viable combinations found: {len(viable_combinations)}")
    
    if viable_combinations:
        # Sort by hurdle rate
        viable_combinations.sort(key=lambda x: x[2])
        
        print(f"\n✅ TOP 15 VIABLE COMBINATIONS:")
        print(f"{'Opp Cost':<10} {'Volatility':<12} {'Hurdle Rate':<12}")
        print("-" * 40)
        
        for opp_cost, vol, hurdle in viable_combinations[:15]:
            print(f"{opp_cost:<10.2f} {vol:<12.1f} {hurdle:<12.3f}")
        
        return viable_combinations
    
    return []

def create_enhanced_solver_with_viable_params(viable_combinations: List[Tuple]) -> List[Dict]:
    """Create solutions using the viable parameter combinations."""
    
    from airdrop_calculator.types import SolverConstraints, AirdropParameters
    from airdrop_calculator.solver import EnhancedZ3Solver
    
    if not viable_combinations:
        print("❌ No viable combinations to work with")
        return []
    
    print(f"\n🚀 Creating solutions with {len(viable_combinations)} viable combinations...")
    print("=" * 70)
    
    solver = EnhancedZ3Solver()
    calculator = AirdropCalculator(solver)
    solutions = []
    
    # Try top 10 viable combinations
    for i, (opp_cost, volatility, expected_hurdle) in enumerate(viable_combinations[:10]):
        print(f"\n--- Attempt {i+1}: OppCost {opp_cost:.2f}%, Vol {volatility:.1f}% ---")
        
        try:
            # Method 1: Soft constraints
            constraints = SolverConstraints(
                min_supply=2_000_000_000,
                max_supply=40_000_000_000,
                min_price=0.001,
                max_price=0.49,
                min_airdrop_percent=25.0,
                max_airdrop_percent=70.0,
                opportunity_cost=opp_cost,
                volatility=volatility,
                gas_cost=20.0,
                campaign_duration=8
            )
            
            objectives = {
                'market_cap': (2_000_000_000, 0.3),
                'profitable_users': (90.0, 0.7)  # High weight on profitability
            }
            
            solution = solver.solve_with_soft_constraints(objectives, constraints)
            
            if solution:
                # Verify hurdle rate
                actual_hurdle = calculator.calculate_hurdle_rate(
                    solution.opportunity_cost, solution.volatility
                )
                
                if actual_hurdle < 5.0:
                    # Analyze profitability
                    profitable_pop = 0.0
                    for segment in calculator.user_segments:
                        avg_capital = (segment.min_capital + segment.max_capital) / 2
                        
                        cliff_value = calculator.calculate_cliff_value(
                            capital=avg_capital,
                            opportunity_cost=solution.opportunity_cost,
                            volatility=solution.volatility,
                            time_months=solution.campaign_duration,
                            gas_cost=solution.gas_cost,
                            num_transactions=segment.avg_transactions
                        )
                        
                        min_tokens = cliff_value / solution.launch_price
                        estimated_allocation = calculator.estimate_user_allocation(
                            capital=avg_capital,
                            allocation_model="quadratic",
                            total_supply=solution.total_supply,
                            airdrop_percent=solution.airdrop_percent
                        )
                        
                        if estimated_allocation >= min_tokens:
                            profitable_pop += segment.population_percent
                    
                    market_cap = solution.total_supply * solution.launch_price
                    
                    solution_data = {
                        "approach": f"viable_params_{i+1}",
                        "parameters": {k: float(v) if hasattr(v, 'item') else v for k, v in solution.__dict__.items()},
                        "hurdle_rate": float(actual_hurdle),
                        "market_cap": float(market_cap),
                        "profitable_population": float(profitable_pop),
                        "invariants": {
                            "price_below_050": solution.launch_price < 0.50,
                            "supply_min_2b": solution.total_supply >= 2_000_000_000,
                            "opp_cost_min_2": solution.opportunity_cost >= 2.0,
                            "airdrop_generous": solution.airdrop_percent >= 25.0,
                            "hurdle_rate_below_5": actual_hurdle < 5.0,
                            "majority_profitable": profitable_pop >= 75.0
                        },
                        "summary": {
                            "price": f"${solution.launch_price:.3f}",
                            "supply": f"{solution.total_supply:,.0f}",
                            "airdrop": f"{solution.airdrop_percent:.1f}%",
                            "hurdle_rate": f"{actual_hurdle:.3f}x",
                            "profitable_users": f"{profitable_pop:.1f}%"
                        }
                    }
                    
                    solutions.append(solution_data)
                    
                    print(f"✅ Solution found:")
                    print(f"   • Price: {solution_data['summary']['price']}")
                    print(f"   • Hurdle Rate: {solution_data['summary']['hurdle_rate']}")
                    print(f"   • Profitable Users: {solution_data['summary']['profitable_users']}")
                    
                else:
                    print(f"❌ Hurdle rate too high: {actual_hurdle:.3f}")
            else:
                print("❌ No solution found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return solutions

def main():
    """Main analysis and solution generation."""
    print("🎯 HURDLE RATE CONSTRAINT ANALYSIS")
    print("=" * 80)
    
    # Step 1: Analyze parameter space
    viable_combinations = analyze_hurdle_rate_space()
    
    # Step 2: Extended search if needed
    if len(viable_combinations) < 10:
        viable_combinations.extend(find_extended_viable_combinations())
    
    # Step 3: Create solutions with viable parameters
    if viable_combinations:
        solutions = create_enhanced_solver_with_viable_params(viable_combinations)
        
        # Save results
        output_file = "hurdle_analysis_solutions.json"
        with open(output_file, 'w') as f:
            json.dump(solutions, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        
        # Display results
        if solutions:
            print(f"\n🏆 SUCCESSFUL SOLUTIONS WITH HURDLE RATE < 5.0:")
            print("=" * 80)
            
            for i, solution in enumerate(solutions, 1):
                all_invariants = all(solution['invariants'].values())
                print(f"\n{i}. {solution['approach'].upper()}")
                print(f"   Price: {solution['summary']['price']} {'✅' if solution['invariants']['price_below_050'] else '❌'}")
                print(f"   Supply: {solution['summary']['supply']} {'✅' if solution['invariants']['supply_min_2b'] else '❌'}")
                print(f"   Airdrop: {solution['summary']['airdrop']} {'✅' if solution['invariants']['airdrop_generous'] else '❌'}")
                print(f"   Hurdle Rate: {solution['summary']['hurdle_rate']} {'✅' if solution['invariants']['hurdle_rate_below_5'] else '❌'}")
                print(f"   Profitable Users: {solution['summary']['profitable_users']} {'✅' if solution['invariants']['majority_profitable'] else '❌'}")
                print(f"   All Constraints: {'✅' if all_invariants else '❌'}")
            
            # Find best solution
            valid_solutions = [s for s in solutions if all(s['invariants'].values())]
            if valid_solutions:
                best = min(valid_solutions, key=lambda x: x['hurdle_rate'])
                print(f"\n🥇 BEST SOLUTION:")
                print(f"   • Approach: {best['approach']}")
                print(f"   • Price: {best['summary']['price']}")
                print(f"   • Hurdle Rate: {best['summary']['hurdle_rate']}")
                print(f"   • Opportunity Cost: {best['parameters']['opportunity_cost']:.2f}%")
                print(f"   • Volatility: {best['parameters']['volatility']:.1f}%")
                print(f"   • Profitable Users: {best['summary']['profitable_users']}")
            
            return solutions
        else:
            print("\n❌ No valid solutions found even with viable parameters")
            print("The constraint hurdle_rate < 5.0 may be mathematically incompatible")
            print("with the other requirements (price < $0.50, profitability, etc.)")
    
    else:
        print("\n❌ No viable parameter combinations found")
        print("Recommendation: Relax hurdle rate constraint to < 6.0 or < 7.0")
    
    return []

if __name__ == "__main__":
    solutions = main()