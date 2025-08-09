#!/usr/bin/env python3
"""
Airdrop Parameter Finder with Automatic Constraint Relaxation

This script finds airdrop parameters given specific constraints and automatically
relaxes constraints if no solution is found initially.
"""

import json
import logging
from typing import Optional, Dict, Any
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints, AirdropParameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_solution_with_relaxation(
    opportunity_cost: float = 5.0,
    min_airdrop_percent: float = 30.0,
    airdrop_certainty: float = 100.0,
    gas_cost: float = 1.0,
    min_duration: int = 3,
    min_price: float = 0.1,
    target_market_cap: float = 500_000_000,
    target_profitable_users: float = 70.0
) -> Optional[AirdropParameters]:
    """
    Find airdrop parameters with automatic constraint relaxation.
    
    Args:
        opportunity_cost: User opportunity cost percentage
        min_airdrop_percent: Minimum airdrop percentage
        airdrop_certainty: Certainty of airdrop happening (%)
        gas_cost: Gas cost per transaction
        min_duration: Minimum campaign duration (months)
        min_price: Minimum launch price
        target_market_cap: Target market capitalization
        target_profitable_users: Target percentage of profitable users
    
    Returns:
        AirdropParameters if solution found, None otherwise
    """
    
    solver = EnhancedZ3Solver()
    
    # Define initial constraints
    initial_constraints = SolverConstraints(
        min_airdrop_percent=min_airdrop_percent,
        max_airdrop_percent=50.0,  # Reasonable upper bound
        opportunity_cost=opportunity_cost,
        gas_cost=gas_cost,
        campaign_duration=min_duration,
        min_price=min_price,
        max_price=10.0,  # Reasonable upper bound
        min_supply=100_000_000,
        max_supply=10_000_000_000
    )
    
    logger.info("="*60)
    logger.info("AIRDROP PARAMETER FINDER")
    logger.info("="*60)
    logger.info(f"Initial Constraints:")
    logger.info(f"  Opportunity Cost: {opportunity_cost}%")
    logger.info(f"  Min Airdrop %: {min_airdrop_percent}%")
    logger.info(f"  Airdrop Certainty: {airdrop_certainty}%")
    logger.info(f"  Gas Cost: ${gas_cost}")
    logger.info(f"  Min Duration: {min_duration} months")
    logger.info(f"  Min Price: ${min_price}")
    logger.info(f"  Target Market Cap: ${target_market_cap:,.0f}")
    logger.info(f"  Target Profitable Users: {target_profitable_users}%")
    logger.info("-"*60)
    
    # Method 1: Try direct nonlinear constraints solver
    logger.info("🔍 Method 1: Trying nonlinear constraints solver...")
    try:
        solution = solver.solve_with_nonlinear_constraints(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            constraints=initial_constraints
        )
        
        if solution:
            logger.info("✅ Solution found with nonlinear constraints!")
            return solution
        else:
            logger.info("❌ No solution found with nonlinear constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with nonlinear solver: {e}")
    
    # Method 2: Try soft constraints solver
    logger.info("🔍 Method 2: Trying soft constraints solver...")
    try:
        objectives = {
            'market_cap': (target_market_cap, 1.0),
            'profitable_users': (target_profitable_users, 0.8)
        }
        
        solution = solver.solve_with_soft_constraints(
            objectives=objectives,
            constraints=initial_constraints
        )
        
        if solution:
            logger.info("✅ Solution found with soft constraints!")
            return solution
        else:
            logger.info("❌ No solution found with soft constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with soft constraints solver: {e}")
    
    # Method 3: Try Kalman-enhanced incremental solver
    logger.info("🔍 Method 3: Trying Kalman-enhanced incremental solver...")
    try:
        solution = solver.solve_incremental_with_kalman(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            initial_constraints=initial_constraints,
            max_iterations=15
        )
        
        if solution:
            logger.info("✅ Solution found with Kalman-enhanced solver!")
            return solution
        else:
            logger.info("❌ No solution found with Kalman solver")
    except Exception as e:
        logger.warning(f"⚠️ Error with Kalman solver: {e}")
    
    # Method 4: Progressive constraint relaxation
    logger.info("🔍 Method 4: Trying progressive constraint relaxation...")
    
    relaxation_levels = [
        # Level 1: Relax airdrop percentage slightly
        (90, SolverConstraints(
            min_airdrop_percent=max(20.0, min_airdrop_percent - 5),
            max_airdrop_percent=60.0,
            opportunity_cost=opportunity_cost,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=max(0.05, min_price - 0.02),
            max_price=15.0
        )),
        
        # Level 2: Relax airdrop percentage more
        (70, SolverConstraints(
            min_airdrop_percent=max(15.0, min_airdrop_percent - 10),
            max_airdrop_percent=70.0,
            opportunity_cost=opportunity_cost,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=max(0.01, min_price - 0.05),
            max_price=20.0
        )),
        
        # Level 3: Significant relaxation
        (50, SolverConstraints(
            min_airdrop_percent=10.0,
            max_airdrop_percent=80.0,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=0.01,
            max_price=50.0
        )),
        
        # Level 4: Maximum relaxation
        (30, SolverConstraints(
            min_airdrop_percent=5.0,
            max_airdrop_percent=90.0,
            min_price=0.001,
            max_price=100.0
        ))
    ]
    
    try:
        solution = solver.solve_incremental_with_relaxation(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            constraint_levels=relaxation_levels
        )
        
        if solution:
            logger.info("✅ Solution found with progressive relaxation!")
            return solution
        else:
            logger.info("❌ No solution found even with maximum relaxation")
    except Exception as e:
        logger.warning(f"⚠️ Error with progressive relaxation: {e}")
    
    # Method 5: Last resort - very relaxed constraints
    logger.info("🔍 Method 5: Last resort with minimal constraints...")
    try:
        minimal_constraints = SolverConstraints(
            min_airdrop_percent=1.0,
            max_airdrop_percent=95.0,
            min_price=0.001,
            max_price=1000.0,
            min_supply=1_000_000,
            max_supply=50_000_000_000
        )
        
        # Try with very relaxed targets
        relaxed_objectives = {
            'market_cap': (target_market_cap * 0.5, 1.0),  # 50% of target
            'profitable_users': (target_profitable_users * 0.7, 0.5)  # 70% of target
        }
        
        solution = solver.solve_with_soft_constraints(
            objectives=relaxed_objectives,
            constraints=minimal_constraints
        )
        
        if solution:
            logger.info("✅ Solution found with minimal constraints!")
            return solution
        else:
            logger.info("❌ No solution found even with minimal constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with minimal constraints: {e}")
    
    logger.error("🚫 Unable to find any solution with all attempted methods")
    return None

def print_solution_analysis(solution: AirdropParameters) -> None:
    """Print detailed analysis of the found solution."""
    
    logger.info("="*60)
    logger.info("SOLUTION ANALYSIS")
    logger.info("="*60)
    
    # Basic parameters
    logger.info("📊 Basic Parameters:")
    logger.info(f"  Total Supply: {solution.total_supply:,.0f} tokens")
    logger.info(f"  Airdrop Percentage: {solution.airdrop_percent:.2f}%")
    logger.info(f"  Launch Price: ${solution.launch_price:.4f}")
    logger.info(f"  Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
    
    # Economic parameters
    logger.info("\n💰 Economic Parameters:")
    logger.info(f"  Opportunity Cost: {solution.opportunity_cost:.2f}%")
    logger.info(f"  Volatility: {solution.volatility:.2f}%")
    logger.info(f"  Beta: {solution.beta:.4f}" if solution.beta else "  Beta: Not calculated")
    logger.info(f"  Hurdle Rate: {solution.hurdle_rate:.4f}" if solution.hurdle_rate else "  Hurdle Rate: Not calculated")
    
    # Operational parameters
    logger.info("\n⚙️ Operational Parameters:")
    logger.info(f"  Gas Cost: ${solution.gas_cost:.2f}")
    logger.info(f"  Campaign Duration: {solution.campaign_duration} months")
    logger.info(f"  Airdrop Certainty: {solution.airdrop_certainty:.1f}%")
    logger.info(f"  Revenue Share: {solution.revenue_share:.1f}%")
    logger.info(f"  Vesting Months: {solution.vesting_months}")
    logger.info(f"  Immediate Unlock: {solution.immediate_unlock:.1f}%")
    
    # Validation checks
    logger.info("\n✅ Validation Checks:")
    
    # Check if beta > 1
    if solution.beta and solution.beta > 1.0:
        logger.info(f"  ✓ Beta > 1: {solution.beta:.4f} (Valid for option pricing)")
    else:
        logger.info(f"  ⚠️ Beta ≤ 1: {solution.beta} (May not be economically viable)")
    
    # Check hurdle rate
    if solution.hurdle_rate and 1.1 <= solution.hurdle_rate <= 10.0:
        logger.info(f"  ✓ Hurdle rate in viable range: {solution.hurdle_rate:.4f}")
    else:
        logger.info(f"  ⚠️ Hurdle rate outside viable range: {solution.hurdle_rate}")
    
    # Check market cap reasonableness
    market_cap = solution.total_supply * solution.launch_price
    if 100_000_000 <= market_cap <= 10_000_000_000:
        logger.info(f"  ✓ Market cap in reasonable range: ${market_cap:,.0f}")
    else:
        logger.info(f"  ⚠️ Market cap may be unrealistic: ${market_cap:,.0f}")
    
    # Check airdrop percentage
    if 5.0 <= solution.airdrop_percent <= 50.0:
        logger.info(f"  ✓ Airdrop percentage reasonable: {solution.airdrop_percent:.2f}%")
    else:
        logger.info(f"  ⚠️ Airdrop percentage may be extreme: {solution.airdrop_percent:.2f}%")

def save_solution(solution: AirdropParameters, filename: str = "found_solution.json") -> None:
    """Save the solution to a JSON file."""
    
    solution_dict = {
        "total_supply": solution.total_supply,
        "airdrop_percent": solution.airdrop_percent,
        "launch_price": solution.launch_price,
        "opportunity_cost": solution.opportunity_cost,
        "volatility": solution.volatility,
        "gas_cost": solution.gas_cost,
        "campaign_duration": solution.campaign_duration,
        "airdrop_certainty": solution.airdrop_certainty,
        "revenue_share": solution.revenue_share,
        "vesting_months": solution.vesting_months,
        "immediate_unlock": solution.immediate_unlock,
        "beta": solution.beta,
        "hurdle_rate": solution.hurdle_rate,
        "market_cap": solution.total_supply * solution.launch_price,
        "airdrop_tokens": solution.total_supply * (solution.airdrop_percent / 100)
    }
    
    with open(filename, 'w') as f:
        json.dump(solution_dict, f, indent=2)
    
    logger.info(f"💾 Solution saved to: {filename}")

def main():
    """Main function to find and analyze the solution."""
    
    # Your specified constraints
    solution = find_solution_with_relaxation(
        opportunity_cost=5.0,
        min_airdrop_percent=30.0,
        airdrop_certainty=100.0,
        gas_cost=1.0,
        min_duration=3,
        min_price=0.1,
        target_market_cap=500_000_000,
        target_profitable_users=70.0
    )
    
    if solution:
        print_solution_analysis(solution)
        save_solution(solution)
        
        logger.info("="*60)
        logger.info("🎉 SUCCESS: Solution found and saved!")
        logger.info("="*60)
        
        return solution
    else:
        logger.error("="*60)
        logger.error("💥 FAILURE: No solution could be found")
        logger.error("="*60)
        logger.error("Consider:")
        logger.error("  - Lowering target market cap")
        logger.error("  - Reducing minimum airdrop percentage")
        logger.error("  - Increasing price flexibility")
        logger.error("  - Adjusting profitability targets")
        
        return None

if __name__ == "__main__":
    main()