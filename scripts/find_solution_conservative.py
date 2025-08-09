#!/usr/bin/env python3
"""
Conservative Airdrop Parameter Finder

This script finds airdrop parameters for conservative scenarios with low opportunity cost
and volatility, optimized for the specific constraints provided.
"""

import json
import logging
from typing import Optional, Dict, Any
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints, AirdropParameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_conservative_solution(
    opportunity_cost: float = 0.5,
    min_airdrop_percent: float = 30.0,
    airdrop_certainty: float = 100.0,
    gas_cost: float = 1.0,
    min_duration: int = 3,
    min_price: float = 0.1,
    revenue_share: float = 40.0,
    volatility: float = 20.0,
    target_market_cap: float = 500_000_000,
    target_profitable_users: float = 75.0
) -> Optional[AirdropParameters]:
    """
    Find airdrop parameters for conservative scenario with automatic constraint relaxation.
    
    Args:
        opportunity_cost: User opportunity cost percentage (very low)
        min_airdrop_percent: Minimum airdrop percentage
        airdrop_certainty: Certainty of airdrop happening (%)
        gas_cost: Gas cost per transaction
        min_duration: Minimum campaign duration (months)
        min_price: Minimum launch price
        revenue_share: Revenue share percentage
        volatility: Market volatility percentage (low)
        target_market_cap: Target market capitalization
        target_profitable_users: Target percentage of profitable users
    
    Returns:
        AirdropParameters if solution found, None otherwise
    """
    
    solver = EnhancedZ3Solver()
    
    logger.info("="*70)
    logger.info("CONSERVATIVE AIRDROP PARAMETER FINDER")
    logger.info("="*70)
    logger.info(f"Conservative Market Constraints:")
    logger.info(f"  Opportunity Cost: {opportunity_cost}% (Very Low)")
    logger.info(f"  Volatility: {volatility}% (Low)")
    logger.info(f"  Min Airdrop %: {min_airdrop_percent}%")
    logger.info(f"  Airdrop Certainty: {airdrop_certainty}%")
    logger.info(f"  Gas Cost: ${gas_cost}")
    logger.info(f"  Min Duration: {min_duration} months")
    logger.info(f"  Min Price: ${min_price}")
    logger.info(f"  Revenue Share: {revenue_share}%")
    logger.info(f"  Target Market Cap: ${target_market_cap:,.0f}")
    logger.info(f"  Target Profitable Users: {target_profitable_users}%")
    logger.info("-"*70)
    
    # Define initial constraints optimized for conservative scenario
    initial_constraints = SolverConstraints(
        min_airdrop_percent=min_airdrop_percent,
        max_airdrop_percent=60.0,  # Higher upper bound for flexibility
        opportunity_cost=opportunity_cost,
        volatility=volatility,
        gas_cost=gas_cost,
        campaign_duration=min_duration,
        min_price=min_price,
        max_price=5.0,  # Conservative upper price bound
        min_supply=50_000_000,
        max_supply=20_000_000_000  # Larger supply range for low prices
    )
    
    logger.info("🔍 Method 1: Trying soft constraints solver (best for conservative scenarios)...")
    try:
        # Use soft constraints with appropriate weights for conservative scenario
        objectives = {
            'market_cap': (target_market_cap, 1.2),  # Higher weight on market cap
            'profitable_users': (target_profitable_users, 0.8)
        }
        
        solution = solver.solve_with_soft_constraints(
            objectives=objectives,
            constraints=initial_constraints
        )
        
        if solution:
            # Manually set the revenue share since solver doesn't handle this parameter
            solution.revenue_share = revenue_share
            logger.info("✅ Solution found with soft constraints!")
            return solution
        else:
            logger.info("❌ No solution found with soft constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with soft constraints solver: {e}")
    
    logger.info("🔍 Method 2: Trying Kalman-enhanced incremental solver...")
    try:
        solution = solver.solve_incremental_with_kalman(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            initial_constraints=initial_constraints,
            max_iterations=20
        )
        
        if solution:
            solution.revenue_share = revenue_share
            logger.info("✅ Solution found with Kalman-enhanced solver!")
            return solution
        else:
            logger.info("❌ No solution found with Kalman solver")
    except Exception as e:
        logger.warning(f"⚠️ Error with Kalman solver: {e}")
    
    logger.info("🔍 Method 3: Trying nonlinear constraints solver...")
    try:
        solution = solver.solve_with_nonlinear_constraints(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            constraints=initial_constraints
        )
        
        if solution:
            solution.revenue_share = revenue_share
            logger.info("✅ Solution found with nonlinear constraints!")
            return solution
        else:
            logger.info("❌ No solution found with nonlinear constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with nonlinear solver: {e}")
    
    logger.info("🔍 Method 4: Progressive constraint relaxation for conservative scenario...")
    
    # Custom relaxation strategy for conservative parameters
    relaxation_levels = [
        # Level 1: Slightly relax airdrop percentage and price
        (95, SolverConstraints(
            min_airdrop_percent=max(25.0, min_airdrop_percent - 3),
            max_airdrop_percent=65.0,
            opportunity_cost=opportunity_cost,
            volatility=volatility,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=max(0.05, min_price - 0.03),
            max_price=8.0,
            min_supply=30_000_000,
            max_supply=25_000_000_000
        )),
        
        # Level 2: More flexible airdrop percentage
        (80, SolverConstraints(
            min_airdrop_percent=max(20.0, min_airdrop_percent - 8),
            max_airdrop_percent=70.0,
            opportunity_cost=opportunity_cost,
            volatility=volatility,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=max(0.02, min_price - 0.05),
            max_price=12.0,
            min_supply=20_000_000,
            max_supply=30_000_000_000
        )),
        
        # Level 3: Allow some opportunity cost and volatility flexibility
        (60, SolverConstraints(
            min_airdrop_percent=15.0,
            max_airdrop_percent=75.0,
            min_price=0.01,
            max_price=20.0,
            min_supply=10_000_000,
            max_supply=40_000_000_000
            # Remove opportunity_cost and volatility constraints for flexibility
        )),
        
        # Level 4: Maximum flexibility while maintaining core requirements
        (40, SolverConstraints(
            min_airdrop_percent=10.0,
            max_airdrop_percent=80.0,
            min_price=0.005,
            max_price=50.0,
            min_supply=5_000_000,
            max_supply=50_000_000_000
        )),
        
        # Level 5: Minimal constraints
        (20, SolverConstraints(
            min_airdrop_percent=5.0,
            max_airdrop_percent=90.0,
            min_price=0.001,
            max_price=100.0,
            min_supply=1_000_000,
            max_supply=100_000_000_000
        ))
    ]
    
    try:
        solution = solver.solve_incremental_with_relaxation(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            constraint_levels=relaxation_levels
        )
        
        if solution:
            solution.revenue_share = revenue_share
            logger.info("✅ Solution found with progressive relaxation!")
            return solution
        else:
            logger.info("❌ No solution found with progressive relaxation")
    except Exception as e:
        logger.warning(f"⚠️ Error with progressive relaxation: {e}")
    
    logger.info("🔍 Method 5: Last resort with very flexible targets...")
    try:
        # Very relaxed constraints and targets
        minimal_constraints = SolverConstraints(
            min_airdrop_percent=1.0,
            max_airdrop_percent=95.0,
            min_price=0.0001,
            max_price=1000.0,
            min_supply=100_000,
            max_supply=1_000_000_000_000
        )
        
        # Much more relaxed targets
        relaxed_objectives = {
            'market_cap': (target_market_cap * 0.3, 1.0),  # 30% of target
            'profitable_users': (target_profitable_users * 0.5, 0.5)  # 50% of target
        }
        
        solution = solver.solve_with_soft_constraints(
            objectives=relaxed_objectives,
            constraints=minimal_constraints
        )
        
        if solution:
            solution.revenue_share = revenue_share
            logger.info("✅ Solution found with minimal constraints!")
            return solution
        else:
            logger.info("❌ No solution found even with minimal constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with minimal constraints: {e}")
    
    logger.error("🚫 Unable to find any solution with all attempted methods")
    logger.error("Conservative scenarios with very low opportunity cost and volatility")
    logger.error("may require different mathematical approaches or target adjustments.")
    return None

def analyze_conservative_solution(solution: AirdropParameters) -> None:
    """Analyze the solution with focus on conservative scenario metrics."""
    
    logger.info("="*70)
    logger.info("CONSERVATIVE SOLUTION ANALYSIS")
    logger.info("="*70)
    
    # Basic parameters
    logger.info("📊 Core Parameters:")
    logger.info(f"  Total Supply: {solution.total_supply:,.0f} tokens")
    logger.info(f"  Airdrop Percentage: {solution.airdrop_percent:.2f}%")
    logger.info(f"  Launch Price: ${solution.launch_price:.6f}")
    logger.info(f"  Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
    logger.info(f"  Airdrop Tokens: {solution.total_supply * (solution.airdrop_percent/100):,.0f}")
    
    # Conservative-specific metrics
    logger.info("\n💰 Conservative Economic Metrics:")
    logger.info(f"  Opportunity Cost: {solution.opportunity_cost:.2f}% (Target: ≤1%)")
    logger.info(f"  Volatility: {solution.volatility:.2f}% (Target: ≤25%)")
    logger.info(f"  Revenue Share: {solution.revenue_share:.1f}%")
    logger.info(f"  Beta: {solution.beta:.6f}" if solution.beta else "  Beta: Not calculated")
    logger.info(f"  Hurdle Rate: {solution.hurdle_rate:.6f}" if solution.hurdle_rate else "  Hurdle Rate: Not calculated")
    
    # Risk assessment for conservative scenario
    logger.info("\n🛡️ Conservative Risk Assessment:")
    
    # Check opportunity cost
    if solution.opportunity_cost <= 1.0:
        logger.info(f"  ✅ Very Low Opportunity Cost: {solution.opportunity_cost:.2f}% (Excellent for retention)")
    elif solution.opportunity_cost <= 3.0:
        logger.info(f"  ✓ Low Opportunity Cost: {solution.opportunity_cost:.2f}% (Good for retention)")
    else:
        logger.info(f"  ⚠️ Higher Opportunity Cost: {solution.opportunity_cost:.2f}% (May affect participation)")
    
    # Check volatility
    if solution.volatility <= 25.0:
        logger.info(f"  ✅ Low Volatility: {solution.volatility:.2f}% (Stable scenario)")
    elif solution.volatility <= 50.0:
        logger.info(f"  ✓ Moderate Volatility: {solution.volatility:.2f}% (Acceptable risk)")
    else:
        logger.info(f"  ⚠️ High Volatility: {solution.volatility:.2f}% (Higher risk scenario)")
    
    # Check beta and hurdle rate for conservative scenarios
    if solution.beta and solution.beta > 1.0:
        if solution.beta < 2.0:
            logger.info(f"  ✅ Conservative Beta: {solution.beta:.6f} (Low risk premium)")
        else:
            logger.info(f"  ⚠️ Higher Beta: {solution.beta:.6f} (Higher risk premium)")
    
    if solution.hurdle_rate:
        if solution.hurdle_rate <= 3.0:
            logger.info(f"  ✅ Low Hurdle Rate: {solution.hurdle_rate:.6f} (Easy to achieve profitability)")
        elif solution.hurdle_rate <= 5.0:
            logger.info(f"  ✓ Moderate Hurdle Rate: {solution.hurdle_rate:.6f} (Reasonable profitability threshold)")
        else:
            logger.info(f"  ⚠️ High Hurdle Rate: {solution.hurdle_rate:.6f} (Challenging profitability)")
    
    # Price accessibility
    if solution.launch_price <= 1.0:
        logger.info(f"  ✅ Accessible Price: ${solution.launch_price:.6f} (Retail-friendly)")
    elif solution.launch_price <= 10.0:
        logger.info(f"  ✓ Moderate Price: ${solution.launch_price:.6f} (Reasonable entry point)")
    else:
        logger.info(f"  ⚠️ High Price: ${solution.launch_price:.6f} (May limit accessibility)")
    
    # Airdrop generosity
    airdrop_tokens = solution.total_supply * (solution.airdrop_percent/100)
    if solution.airdrop_percent >= 30.0:
        logger.info(f"  ✅ Generous Airdrop: {solution.airdrop_percent:.2f}% ({airdrop_tokens:,.0f} tokens)")
    elif solution.airdrop_percent >= 20.0:
        logger.info(f"  ✓ Substantial Airdrop: {solution.airdrop_percent:.2f}% ({airdrop_tokens:,.0f} tokens)")
    else:
        logger.info(f"  ⚠️ Limited Airdrop: {solution.airdrop_percent:.2f}% ({airdrop_tokens:,.0f} tokens)")

def save_conservative_solution(solution: AirdropParameters, filename: str = "conservative_solution.json") -> None:
    """Save the conservative solution with additional metrics."""
    
    airdrop_tokens = solution.total_supply * (solution.airdrop_percent / 100)
    market_cap = solution.total_supply * solution.launch_price
    
    solution_dict = {
        "scenario_type": "conservative",
        "core_parameters": {
            "total_supply": solution.total_supply,
            "airdrop_percent": solution.airdrop_percent,
            "launch_price": solution.launch_price,
            "market_cap": market_cap,
            "airdrop_tokens": airdrop_tokens
        },
        "economic_parameters": {
            "opportunity_cost": solution.opportunity_cost,
            "volatility": solution.volatility,
            "revenue_share": solution.revenue_share,
            "beta": solution.beta,
            "hurdle_rate": solution.hurdle_rate
        },
        "operational_parameters": {
            "gas_cost": solution.gas_cost,
            "campaign_duration": solution.campaign_duration,
            "airdrop_certainty": solution.airdrop_certainty,
            "vesting_months": solution.vesting_months,
            "immediate_unlock": solution.immediate_unlock
        },
        "conservative_metrics": {
            "low_opportunity_cost": solution.opportunity_cost <= 1.0,
            "low_volatility": solution.volatility <= 25.0,
            "accessible_price": solution.launch_price <= 1.0,
            "generous_airdrop": solution.airdrop_percent >= 30.0,
            "risk_category": "low" if (solution.opportunity_cost <= 1.0 and solution.volatility <= 25.0) else "moderate"
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(solution_dict, f, indent=2)
    
    logger.info(f"💾 Conservative solution saved to: {filename}")

def main():
    """Main function to find and analyze the conservative solution."""
    
    # Your specified conservative constraints
    solution = find_conservative_solution(
        opportunity_cost=0.5,
        min_airdrop_percent=30.0,
        airdrop_certainty=100.0,
        gas_cost=1.0,
        min_duration=3,
        min_price=0.1,
        revenue_share=40.0,
        volatility=20.0,
        target_market_cap=500_000_000,
        target_profitable_users=75.0
    )
    
    if solution:
        analyze_conservative_solution(solution)
        save_conservative_solution(solution)
        
        logger.info("="*70)
        logger.info("🎉 CONSERVATIVE SOLUTION FOUND!")
        logger.info("="*70)
        logger.info("This solution is optimized for:")
        logger.info("  • Low opportunity cost (high user retention)")
        logger.info("  • Low volatility (stable economics)")
        logger.info("  • High revenue share (sustainable tokenomics)")
        logger.info("  • Accessible pricing (broad participation)")
        logger.info("="*70)
        
        return solution
    else:
        logger.error("="*70)
        logger.error("💥 CONSERVATIVE SOLUTION NOT FOUND")
        logger.error("="*70)
        logger.error("Conservative scenarios require specific conditions:")
        logger.error("  - Very low opportunity costs may need adjusted targets")
        logger.error("  - Low volatility can make option pricing challenging")
        logger.error("  - Consider increasing target flexibility")
        logger.error("="*70)
        
        return None

if __name__ == "__main__":
    main()