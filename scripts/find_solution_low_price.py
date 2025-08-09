#!/usr/bin/env python3
"""
Low Price Airdrop Parameter Finder

This script finds airdrop parameters for conservative scenarios with strict price limits,
using non-Kalman solvers only.
"""

import json
import logging
from typing import Optional, Dict, Any
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints, AirdropParameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_low_price_solution(
    opportunity_cost: float = 0.5,
    min_airdrop_percent: float = 30.0,
    airdrop_certainty: float = 100.0,
    gas_cost: float = 1.0,
    min_duration: int = 3,
    min_price: float = 0.1,
    max_price: float = 0.50,
    revenue_share: float = 40.0,
    volatility: float = 20.0,
    target_market_cap: float = 500_000_000,
    target_profitable_users: float = 75.0
) -> Optional[AirdropParameters]:
    """
    Find airdrop parameters with strict price limits using non-Kalman solvers.
    
    Args:
        opportunity_cost: User opportunity cost percentage (very low)
        min_airdrop_percent: Minimum airdrop percentage
        airdrop_certainty: Certainty of airdrop happening (%)
        gas_cost: Gas cost per transaction
        min_duration: Minimum campaign duration (months)
        min_price: Minimum launch price
        max_price: Maximum launch price (strict limit)
        revenue_share: Revenue share percentage
        volatility: Market volatility percentage (low)
        target_market_cap: Target market capitalization
        target_profitable_users: Target percentage of profitable users
    
    Returns:
        AirdropParameters if solution found, None otherwise
    """
    
    solver = EnhancedZ3Solver()
    
    logger.info("="*70)
    logger.info("LOW PRICE AIRDROP PARAMETER FINDER")
    logger.info("="*70)
    logger.info(f"Strict Price Limit Constraints:")
    logger.info(f"  Opportunity Cost: {opportunity_cost}%")
    logger.info(f"  Volatility: {volatility}%")
    logger.info(f"  Min Airdrop %: {min_airdrop_percent}%")
    logger.info(f"  Airdrop Certainty: {airdrop_certainty}%")
    logger.info(f"  Gas Cost: ${gas_cost}")
    logger.info(f"  Min Duration: {min_duration} months")
    logger.info(f"  Price Range: ${min_price} - ${max_price} (STRICT)")
    logger.info(f"  Revenue Share: {revenue_share}%")
    logger.info(f"  Target Market Cap: ${target_market_cap:,.0f}")
    logger.info(f"  Target Profitable Users: {target_profitable_users}%")
    logger.info(f"  Solvers: Non-Kalman only")
    logger.info("-"*70)
    
    # Calculate required supply range for price constraints
    min_supply_for_price = target_market_cap / max_price  # Minimum supply needed for max price
    max_supply_for_price = target_market_cap / min_price  # Maximum supply allowed for min price
    
    logger.info(f"📊 Price-based supply constraints:")
    logger.info(f"  Min supply for max price ${max_price}: {min_supply_for_price:,.0f}")
    logger.info(f"  Max supply for min price ${min_price}: {max_supply_for_price:,.0f}")
    
    # Define initial constraints optimized for low price scenario
    initial_constraints = SolverConstraints(
        min_airdrop_percent=min_airdrop_percent,
        max_airdrop_percent=50.0,
        opportunity_cost=opportunity_cost,
        volatility=volatility,
        gas_cost=gas_cost,
        campaign_duration=min_duration,
        min_price=min_price,
        max_price=max_price,  # Strict price limit
        min_supply=max(100_000_000, int(min_supply_for_price * 0.8)),  # 80% buffer
        max_supply=min(50_000_000_000, int(max_supply_for_price * 1.2))  # 20% buffer
    )
    
    logger.info("🔍 Method 1: Trying soft constraints solver...")
    try:
        # Use soft constraints with higher weight on market cap to find large supply solutions
        objectives = {
            'market_cap': (target_market_cap, 1.5),  # Higher weight to drive toward target
            'profitable_users': (target_profitable_users, 0.7)
        }
        
        solution = solver.solve_with_soft_constraints(
            objectives=objectives,
            constraints=initial_constraints
        )
        
        if solution and solution.launch_price <= max_price:
            solution.revenue_share = revenue_share
            logger.info(f"✅ Solution found with soft constraints! Price: ${solution.launch_price:.4f}")
            return solution
        elif solution:
            logger.info(f"❌ Solution price ${solution.launch_price:.4f} exceeds limit ${max_price}")
        else:
            logger.info("❌ No solution found with soft constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with soft constraints solver: {e}")
    
    logger.info("🔍 Method 2: Trying nonlinear constraints solver...")
    try:
        solution = solver.solve_with_nonlinear_constraints(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            constraints=initial_constraints
        )
        
        if solution and solution.launch_price <= max_price:
            solution.revenue_share = revenue_share
            logger.info(f"✅ Solution found with nonlinear constraints! Price: ${solution.launch_price:.4f}")
            return solution
        elif solution:
            logger.info(f"❌ Solution price ${solution.launch_price:.4f} exceeds limit ${max_price}")
        else:
            logger.info("❌ No solution found with nonlinear constraints")
    except Exception as e:
        logger.warning(f"⚠️ Error with nonlinear solver: {e}")
    
    logger.info("🔍 Method 3: Progressive constraint relaxation with price limits...")
    
    # Custom relaxation strategy that maintains price limits
    relaxation_levels = [
        # Level 1: Relax airdrop percentage slightly, keep price strict
        (95, SolverConstraints(
            min_airdrop_percent=max(25.0, min_airdrop_percent - 3),
            max_airdrop_percent=55.0,
            opportunity_cost=opportunity_cost,
            volatility=volatility,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=min_price,
            max_price=max_price,  # Keep strict
            min_supply=max(200_000_000, int(min_supply_for_price * 0.7)),
            max_supply=min(100_000_000_000, int(max_supply_for_price * 1.5))
        )),
        
        # Level 2: More flexible airdrop percentage, keep price strict
        (80, SolverConstraints(
            min_airdrop_percent=max(20.0, min_airdrop_percent - 8),
            max_airdrop_percent=60.0,
            opportunity_cost=opportunity_cost,
            volatility=volatility,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=min_price,
            max_price=max_price,  # Keep strict
            min_supply=max(500_000_000, int(min_supply_for_price * 0.6)),
            max_supply=min(200_000_000_000, int(max_supply_for_price * 2.0))
        )),
        
        # Level 3: Allow some parameter flexibility but maintain price limit
        (60, SolverConstraints(
            min_airdrop_percent=15.0,
            max_airdrop_percent=70.0,
            gas_cost=gas_cost,
            campaign_duration=min_duration,
            min_price=min_price,
            max_price=max_price,  # Keep strict
            min_supply=1_000_000_000,  # Large supply for low price
            max_supply=500_000_000_000
        )),
        
        # Level 4: Maximum airdrop flexibility, strict price
        (40, SolverConstraints(
            min_airdrop_percent=10.0,
            max_airdrop_percent=80.0,
            min_price=min_price,
            max_price=max_price,  # Keep strict
            min_supply=2_000_000_000,  # Very large supply
            max_supply=1_000_000_000_000
        )),
        
        # Level 5: Minimal constraints but strict price
        (20, SolverConstraints(
            min_airdrop_percent=5.0,
            max_airdrop_percent=90.0,
            min_price=min_price,
            max_price=max_price,  # Always strict
            min_supply=5_000_000_000,
            max_supply=2_000_000_000_000
        ))
    ]
    
    try:
        solution = solver.solve_incremental_with_relaxation(
            target_market_cap=target_market_cap,
            target_profitable_users=target_profitable_users,
            constraint_levels=relaxation_levels
        )
        
        if solution and solution.launch_price <= max_price:
            solution.revenue_share = revenue_share
            logger.info(f"✅ Solution found with progressive relaxation! Price: ${solution.launch_price:.4f}")
            return solution
        elif solution:
            logger.info(f"❌ Solution price ${solution.launch_price:.4f} exceeds limit ${max_price}")
        else:
            logger.info("❌ No solution found with progressive relaxation")
    except Exception as e:
        logger.warning(f"⚠️ Error with progressive relaxation: {e}")
    
    logger.info("🔍 Method 4: Lower market cap targets with strict price limits...")
    try:
        # Try with significantly lower market cap targets to enable lower prices
        lower_targets = [0.8, 0.6, 0.4, 0.2]  # 80%, 60%, 40%, 20% of original target
        
        for factor in lower_targets:
            adjusted_target = target_market_cap * factor
            logger.info(f"  Trying market cap target: ${adjusted_target:,.0f}")
            
            # Very flexible constraints except for price
            flexible_constraints = SolverConstraints(
                min_airdrop_percent=10.0,
                max_airdrop_percent=80.0,
                min_price=min_price,
                max_price=max_price,  # Always strict
                min_supply=int(adjusted_target / max_price),  # Minimum supply for this target
                max_supply=int(adjusted_target / min_price) * 2  # Allow larger supply
            )
            
            # Try soft constraints with adjusted target
            objectives = {
                'market_cap': (adjusted_target, 1.0),
                'profitable_users': (target_profitable_users * 0.7, 0.5)  # Lower user target too
            }
            
            solution = solver.solve_with_soft_constraints(
                objectives=objectives,
                constraints=flexible_constraints
            )
            
            if solution and solution.launch_price <= max_price:
                solution.revenue_share = revenue_share
                logger.info(f"✅ Solution found with adjusted market cap! Price: ${solution.launch_price:.4f}")
                logger.info(f"  Market cap: ${solution.total_supply * solution.launch_price:,.0f}")
                return solution
            elif solution:
                logger.info(f"  Price ${solution.launch_price:.4f} still exceeds limit")
        
        logger.info("❌ No solution found even with lower market cap targets")
    except Exception as e:
        logger.warning(f"⚠️ Error with lower market cap approach: {e}")
    
    logger.info("🔍 Method 5: Minimal viable product approach...")
    try:
        # Last resort: Find any economically viable solution under price limit
        mvp_constraints = SolverConstraints(
            min_airdrop_percent=1.0,
            max_airdrop_percent=95.0,
            min_price=0.01,  # Very low minimum
            max_price=max_price,  # Strict maximum
            min_supply=10_000_000_000,  # Large supply for low price
            max_supply=5_000_000_000_000
        )
        
        # Very relaxed objectives
        mvp_objectives = {
            'market_cap': (target_market_cap * 0.1, 1.0),  # 10% of target
            'profitable_users': (30.0, 0.3)  # Much lower user target
        }
        
        solution = solver.solve_with_soft_constraints(
            objectives=mvp_objectives,
            constraints=mvp_constraints
        )
        
        if solution and solution.launch_price <= max_price:
            solution.revenue_share = revenue_share
            logger.info(f"✅ MVP solution found! Price: ${solution.launch_price:.4f}")
            logger.info(f"  Market cap: ${solution.total_supply * solution.launch_price:,.0f}")
            return solution
        elif solution:
            logger.info(f"❌ Even MVP solution price ${solution.launch_price:.4f} exceeds limit")
        else:
            logger.info("❌ No MVP solution found")
    except Exception as e:
        logger.warning(f"⚠️ Error with MVP approach: {e}")
    
    logger.error("🚫 Unable to find any solution under the strict price limit")
    logger.error(f"The price limit of ${max_price} may be too restrictive for the target market cap")
    logger.error("Consider:")
    logger.error(f"  - Increasing max price above ${max_price}")
    logger.error(f"  - Lowering target market cap below ${target_market_cap:,.0f}")
    logger.error(f"  - Accepting higher supply (>10B tokens) for lower prices")
    return None

def analyze_low_price_solution(solution: AirdropParameters, max_price: float) -> None:
    """Analyze the solution with focus on low price scenario metrics."""
    
    logger.info("="*70)
    logger.info("LOW PRICE SOLUTION ANALYSIS")
    logger.info("="*70)
    
    # Basic parameters
    logger.info("📊 Core Parameters:")
    logger.info(f"  Total Supply: {solution.total_supply:,.0f} tokens")
    logger.info(f"  Airdrop Percentage: {solution.airdrop_percent:.2f}%")
    logger.info(f"  Launch Price: ${solution.launch_price:.6f}")
    logger.info(f"  Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
    logger.info(f"  Airdrop Tokens: {solution.total_supply * (solution.airdrop_percent/100):,.0f}")
    
    # Price compliance
    logger.info("\n💰 Price Compliance:")
    if solution.launch_price <= max_price:
        logger.info(f"  ✅ Price within limit: ${solution.launch_price:.6f} ≤ ${max_price}")
        price_margin = ((max_price - solution.launch_price) / max_price) * 100
        logger.info(f"  💡 Price margin: {price_margin:.1f}% below limit")
    else:
        logger.info(f"  ❌ Price exceeds limit: ${solution.launch_price:.6f} > ${max_price}")
    
    # Low price benefits analysis
    logger.info("\n🎯 Low Price Benefits:")
    
    # Accessibility
    if solution.launch_price <= 0.1:
        logger.info(f"  ✅ Extremely Accessible: ${solution.launch_price:.6f} (Mass market)")
    elif solution.launch_price <= 0.3:
        logger.info(f"  ✅ Very Accessible: ${solution.launch_price:.6f} (Broad retail)")
    elif solution.launch_price <= 0.5:
        logger.info(f"  ✓ Accessible: ${solution.launch_price:.6f} (Retail friendly)")
    else:
        logger.info(f"  ⚠️ Limited Accessibility: ${solution.launch_price:.6f}")
    
    # Supply implications
    if solution.total_supply >= 10_000_000_000:
        logger.info(f"  📈 High Supply Strategy: {solution.total_supply:,.0f} tokens (Low price enabler)")
    elif solution.total_supply >= 1_000_000_000:
        logger.info(f"  📊 Large Supply: {solution.total_supply:,.0f} tokens (Moderate price)")
    else:
        logger.info(f"  📉 Standard Supply: {solution.total_supply:,.0f} tokens")
    
    # Economic viability
    logger.info("\n⚖️ Economic Assessment:")
    logger.info(f"  Opportunity Cost: {solution.opportunity_cost:.2f}%")
    logger.info(f"  Volatility: {solution.volatility:.2f}%")
    logger.info(f"  Revenue Share: {solution.revenue_share:.1f}%")
    logger.info(f"  Beta: {solution.beta:.6f}" if solution.beta else "  Beta: Not calculated")
    logger.info(f"  Hurdle Rate: {solution.hurdle_rate:.6f}" if solution.hurdle_rate else "  Hurdle Rate: Not calculated")
    
    # Token distribution analysis
    airdrop_tokens = solution.total_supply * (solution.airdrop_percent / 100)
    market_cap = solution.total_supply * solution.launch_price
    airdrop_value = airdrop_tokens * solution.launch_price
    
    logger.info("\n🎁 Airdrop Analysis:")
    logger.info(f"  Total Airdrop Value: ${airdrop_value:,.0f}")
    logger.info(f"  Average per 1% user base: ${airdrop_value / 100:,.0f}")
    logger.info(f"  Tokens per dollar invested: {1/solution.launch_price:,.2f}")

def save_low_price_solution(solution: AirdropParameters, max_price: float, filename: str = "low_price_solution.json") -> None:
    """Save the low price solution with additional metrics."""
    
    airdrop_tokens = solution.total_supply * (solution.airdrop_percent / 100)
    market_cap = solution.total_supply * solution.launch_price
    
    solution_dict = {
        "scenario_type": "low_price",
        "price_compliance": {
            "launch_price": solution.launch_price,
            "max_price_limit": max_price,
            "within_limit": solution.launch_price <= max_price,
            "price_margin_percent": ((max_price - solution.launch_price) / max_price) * 100 if solution.launch_price <= max_price else None
        },
        "core_parameters": {
            "total_supply": solution.total_supply,
            "airdrop_percent": solution.airdrop_percent,
            "launch_price": solution.launch_price,
            "market_cap": market_cap,
            "airdrop_tokens": airdrop_tokens,
            "airdrop_value": airdrop_tokens * solution.launch_price
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
        "accessibility_metrics": {
            "extremely_accessible": solution.launch_price <= 0.1,
            "very_accessible": solution.launch_price <= 0.3,
            "accessible": solution.launch_price <= 0.5,
            "high_supply_strategy": solution.total_supply >= 10_000_000_000,
            "tokens_per_dollar": 1 / solution.launch_price
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(solution_dict, f, indent=2)
    
    logger.info(f"💾 Low price solution saved to: {filename}")

def main():
    """Main function to find and analyze the low price solution."""
    
    # Your specified constraints with strict price limit
    max_price_limit = 0.50
    
    solution = find_low_price_solution(
        opportunity_cost=0.5,
        min_airdrop_percent=30.0,
        airdrop_certainty=100.0,
        gas_cost=1.0,
        min_duration=3,
        min_price=0.1,
        max_price=max_price_limit,  # Strict limit
        revenue_share=40.0,
        volatility=20.0,
        target_market_cap=500_000_000,
        target_profitable_users=75.0
    )
    
    if solution:
        analyze_low_price_solution(solution, max_price_limit)
        save_low_price_solution(solution, max_price_limit)
        
        logger.info("="*70)
        logger.info("🎉 LOW PRICE SOLUTION FOUND!")
        logger.info("="*70)
        logger.info(f"✅ Price: ${solution.launch_price:.6f} (within ${max_price_limit} limit)")
        logger.info(f"✅ Supply: {solution.total_supply:,.0f} tokens")
        logger.info(f"✅ Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
        logger.info(f"✅ Accessibility: High (retail-friendly pricing)")
        logger.info("="*70)
        
        return solution
    else:
        logger.error("="*70)
        logger.error("💥 NO LOW PRICE SOLUTION FOUND")
        logger.error("="*70)
        logger.error(f"Unable to find solution under ${max_price_limit} price limit")
        logger.error("Recommendations:")
        logger.error(f"  1. Increase price limit above ${max_price_limit}")
        logger.error("  2. Lower target market cap significantly")
        logger.error("  3. Accept very high token supply (>50B tokens)")
        logger.error("  4. Reduce minimum airdrop percentage")
        logger.error("="*70)
        
        return None

if __name__ == "__main__":
    main()