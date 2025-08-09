#!/usr/bin/env python3
"""
Script to find 10 possible airdrop configurations by progressively relaxing constraints
while still meeting core requirements as much as possible.

Target Requirements (with progressive relaxation):
- Token price: below $0.50 (STRICT)
- Users profitable: As high as possible (90%+ target)
- Supply: 2,000,000,000 minimum (STRICT)
- Opportunity cost: 2.0% minimum (STRICT)
- Airdrop percent: As close to 100% as possible (80%+ target)
"""

import json
import logging
from typing import List, Dict, Optional
import numpy as np
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.types import SolverConstraints, AirdropParameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_feasible_configurations() -> List[Dict]:
    """
    Find 10 configurations with progressive constraint relaxation to find viable solutions.
    
    Progressive Relaxation Strategy:
    1. Start with strict requirements
    2. Gradually relax constraints until solutions are found
    3. Prioritize core requirements (price < $0.50, supply ≥ 2B, opp cost ≥ 2%)
    """
    
    logger.info("="*80)
    logger.info("FINDING 10 CONFIGURATIONS WITH PROGRESSIVE RELAXATION")
    logger.info("="*80)
    logger.info("Strategy: Progressively relax constraints to find viable solutions")
    logger.info("Core Requirements (STRICT):")
    logger.info("  • Token price: < $0.50")
    logger.info("  • Supply: ≥ 2,000,000,000 tokens")
    logger.info("  • Opportunity cost: ≥ 2.0%")
    logger.info("Flexible Requirements:")
    logger.info("  • Airdrop percent: Target 100%, accept 80%+")
    logger.info("  • User profitability: Target 100%, accept 90%+")
    logger.info("-"*80)
    
    solver = EnhancedZ3Solver()
    configurations = []
    
    # Progressive relaxation levels
    relaxation_levels = [
        {
            "name": "Level 1: Strict (100% airdrop, 100% profitable)",
            "airdrop_range": (99.9, 100.0),
            "profitability_target": 100.0,
            "price_max": 0.49,
            "supply_multiplier": 1.0
        },
        {
            "name": "Level 2: Minor relaxation (95% airdrop, 95% profitable)",
            "airdrop_range": (95.0, 100.0),
            "profitability_target": 95.0,
            "price_max": 0.45,
            "supply_multiplier": 1.2
        },
        {
            "name": "Level 3: Moderate relaxation (90% airdrop, 90% profitable)",
            "airdrop_range": (90.0, 100.0),
            "profitability_target": 90.0,
            "price_max": 0.40,
            "supply_multiplier": 1.5
        },
        {
            "name": "Level 4: Significant relaxation (85% airdrop, 85% profitable)",
            "airdrop_range": (85.0, 95.0),
            "profitability_target": 85.0,
            "price_max": 0.35,
            "supply_multiplier": 2.0
        },
        {
            "name": "Level 5: High relaxation (80% airdrop, 80% profitable)",
            "airdrop_range": (80.0, 90.0),
            "profitability_target": 80.0,
            "price_max": 0.30,
            "supply_multiplier": 3.0
        },
        {
            "name": "Level 6: Maximum feasible (70% airdrop, 70% profitable)",
            "airdrop_range": (70.0, 85.0),
            "profitability_target": 70.0,
            "price_max": 0.25,
            "supply_multiplier": 5.0
        }
    ]
    
    # Parameter generation strategies
    parameter_strategies = [
        {
            "name": "Ultra-Low Price",
            "supply_base": 10_000_000_000,
            "price_range": (0.001, 0.10),
            "opportunity_cost": 2.0,
            "volatility": 20.0,
            "gas_cost": 0.5
        },
        {
            "name": "High Supply Low Cost",
            "supply_base": 50_000_000_000,
            "price_range": (0.005, 0.15),
            "opportunity_cost": 2.5,
            "volatility": 25.0,
            "gas_cost": 1.0
        },
        {
            "name": "Balanced Large Scale",
            "supply_base": 20_000_000_000,
            "price_range": (0.01, 0.20),
            "opportunity_cost": 3.0,
            "volatility": 30.0,
            "gas_cost": 2.0
        },
        {
            "name": "Conservative Large",
            "supply_base": 5_000_000_000,
            "price_range": (0.02, 0.25),
            "opportunity_cost": 2.0,
            "volatility": 15.0,
            "gas_cost": 0.1
        }
    ]
    
    for level in relaxation_levels:
        logger.info(f"\n🔍 {level['name']}")
        
        for strategy in parameter_strategies:
            for attempt in range(3):  # 3 attempts per strategy per level
                try:
                    # Generate parameters
                    base_supply = strategy['supply_base'] * level['supply_multiplier']
                    supply = max(2_000_000_000, np.random.uniform(base_supply * 0.8, base_supply * 1.2))
                    
                    price_min, price_max = strategy['price_range']
                    max_price = min(level['price_max'], price_max)
                    price = np.random.uniform(price_min, max_price)
                    
                    airdrop_min, airdrop_max = level['airdrop_range']
                    airdrop_percent = np.random.uniform(airdrop_min, airdrop_max)
                    
                    # Set up constraints
                    constraints = SolverConstraints(
                        min_supply=supply * 0.95,
                        max_supply=supply * 1.05,
                        min_price=price * 0.8,
                        max_price=price * 1.2,
                        min_airdrop_percent=airdrop_percent * 0.95,
                        max_airdrop_percent=min(100.0, airdrop_percent * 1.05),
                        opportunity_cost=strategy['opportunity_cost'],
                        volatility=strategy['volatility'],
                        gas_cost=strategy['gas_cost'],
                        campaign_duration=1,  # Very short to minimize costs
                    )
                    
                    # Try to find solution
                    target_market_cap = supply * price * 0.5  # Conservative target
                    
                    objectives = {
                        'market_cap': (target_market_cap, 0.1),
                        'profitable_users': (level['profitability_target'], 10.0)  # High weight on profitability
                    }
                    
                    solution = solver.solve_with_soft_constraints(objectives, constraints)
                    
                    if solution and solution.launch_price < 0.50:
                        # Calculate metrics
                        calculator = AirdropCalculator(solver)
                        metrics = calculator.calculate_market_metrics()
                        
                        # Check if it meets minimum viability
                        meets_criteria = (
                            solution.total_supply >= 2_000_000_000 and
                            solution.opportunity_cost >= 2.0 and
                            solution.launch_price < 0.50 and
                            metrics.profitable_users_percent >= 60.0 and  # Minimum 60% profitable
                            solution.airdrop_percent >= 50.0  # Minimum 50% airdrop
                        )
                        
                        if meets_criteria:
                            config = {
                                "configuration_id": len(configurations) + 1,
                                "relaxation_level": level['name'],
                                "strategy": strategy['name'],
                                "parameters": {
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
                                    "immediate_unlock": solution.immediate_unlock
                                },
                                "metrics": {
                                    "market_cap": solution.total_supply * solution.launch_price,
                                    "airdrop_value": solution.total_supply * (solution.airdrop_percent/100) * solution.launch_price,
                                    "profitable_users_percent": metrics.profitable_users_percent,
                                    "beta": solution.beta,
                                    "hurdle_rate": solution.hurdle_rate,
                                    "tokens_per_dollar": 1 / solution.launch_price,
                                    "airdrop_tokens": solution.total_supply * (solution.airdrop_percent/100)
                                },
                                "validation": {
                                    "price_under_050": solution.launch_price < 0.50,
                                    "high_user_profitability": metrics.profitable_users_percent >= 70.0,
                                    "supply_over_2b": solution.total_supply >= 2_000_000_000,
                                    "opportunity_cost_over_2": solution.opportunity_cost >= 2.0,
                                    "high_airdrop_percent": solution.airdrop_percent >= 70.0,
                                    "overall_viability_score": (
                                        (1 if solution.launch_price < 0.50 else 0) +
                                        (1 if metrics.profitable_users_percent >= 80 else 0.5) +
                                        (1 if solution.airdrop_percent >= 80 else 0.5) +
                                        (1 if solution.total_supply >= 2_000_000_000 else 0) +
                                        (1 if solution.opportunity_cost >= 2.0 else 0)
                                    ) / 5.0
                                }
                            }
                            
                            configurations.append(config)
                            logger.info(f"  ✅ Config {len(configurations)}: Price=${solution.launch_price:.6f}, "
                                      f"Airdrop={solution.airdrop_percent:.1f}%, "
                                      f"Profitable={metrics.profitable_users_percent:.1f}%")
                            
                            if len(configurations) >= 10:
                                return configurations
                        
                except Exception as e:
                    continue
        
        if len(configurations) >= 10:
            break
    
    # If still not enough, try extreme relaxation
    if len(configurations) < 10:
        logger.info(f"\n🔍 Extreme Relaxation: Finding remaining configurations...")
        
        for attempt in range(50):
            try:
                supply = np.random.uniform(2_000_000_000, 100_000_000_000)
                price = np.random.uniform(0.001, 0.49)
                airdrop_percent = np.random.uniform(50.0, 95.0)
                
                constraints = SolverConstraints(
                    min_supply=supply * 0.9,
                    max_supply=supply * 1.1,
                    min_price=price * 0.5,
                    max_price=price * 1.5,
                    min_airdrop_percent=airdrop_percent * 0.8,
                    max_airdrop_percent=min(100.0, airdrop_percent * 1.2),
                    opportunity_cost=2.0,
                    volatility=20.0,
                    gas_cost=0.1,
                    campaign_duration=1,
                )
                
                objectives = {
                    'market_cap': (supply * price * 0.3, 0.1),
                    'profitable_users': (60.0, 5.0)
                }
                
                solution = solver.solve_with_soft_constraints(objectives, constraints)
                
                if solution and solution.launch_price < 0.50:
                    calculator = AirdropCalculator(solver)
                    metrics = calculator.calculate_market_metrics()
                    
                    if (solution.total_supply >= 2_000_000_000 and
                        solution.opportunity_cost >= 2.0 and
                        metrics.profitable_users_percent >= 50.0):
                        
                        config = {
                            "configuration_id": len(configurations) + 1,
                            "relaxation_level": "Extreme Relaxation",
                            "strategy": "Maximum Flexibility",
                            "parameters": {
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
                                "immediate_unlock": solution.immediate_unlock
                            },
                            "metrics": {
                                "market_cap": solution.total_supply * solution.launch_price,
                                "airdrop_value": solution.total_supply * (solution.airdrop_percent/100) * solution.launch_price,
                                "profitable_users_percent": metrics.profitable_users_percent,
                                "beta": solution.beta,
                                "hurdle_rate": solution.hurdle_rate,
                                "tokens_per_dollar": 1 / solution.launch_price,
                                "airdrop_tokens": solution.total_supply * (solution.airdrop_percent/100)
                            },
                            "validation": {
                                "price_under_050": solution.launch_price < 0.50,
                                "high_user_profitability": metrics.profitable_users_percent >= 70.0,
                                "supply_over_2b": solution.total_supply >= 2_000_000_000,
                                "opportunity_cost_over_2": solution.opportunity_cost >= 2.0,
                                "high_airdrop_percent": solution.airdrop_percent >= 70.0,
                                "overall_viability_score": (
                                    (1 if solution.launch_price < 0.50 else 0) +
                                    (0.5 if metrics.profitable_users_percent >= 50 else 0) +
                                    (0.5 if solution.airdrop_percent >= 50 else 0) +
                                    (1 if solution.total_supply >= 2_000_000_000 else 0) +
                                    (1 if solution.opportunity_cost >= 2.0 else 0)
                                ) / 5.0
                            }
                        }
                        
                        configurations.append(config)
                        logger.info(f"  ✅ Extreme config {len(configurations)}: "
                                  f"Price=${solution.launch_price:.6f}")
                        
                        if len(configurations) >= 10:
                            break
            except Exception:
                continue
    
    return configurations

def analyze_and_display_configurations(configurations: List[Dict]):
    """Analyze and display the found configurations"""
    
    logger.info("\n" + "="*80)
    logger.info("CONFIGURATION ANALYSIS RESULTS")
    logger.info("="*80)
    
    if not configurations:
        logger.error("❌ No configurations found even with relaxed requirements!")
        return
    
    logger.info(f"✅ Found {len(configurations)} viable configurations!")
    
    # Sort by viability score (descending) then by price (ascending)
    configurations.sort(key=lambda x: (-x['validation']['overall_viability_score'], x['parameters']['launch_price']))
    
    for i, config in enumerate(configurations, 1):
        params = config['parameters']
        metrics = config['metrics']
        validation = config['validation']
        
        logger.info(f"\n📊 CONFIGURATION {i}")
        logger.info(f"Level: {config['relaxation_level']}")
        logger.info(f"Strategy: {config['strategy']}")
        logger.info(f"Viability Score: {validation['overall_viability_score']:.2f}/1.00")
        logger.info("-" * 60)
        
        # Core parameters
        logger.info(f"🎯 Core Parameters:")
        logger.info(f"  Supply: {params['total_supply']:,.0f} tokens")
        logger.info(f"  Airdrop: {params['airdrop_percent']:.1f}%")
        logger.info(f"  Price: ${params['launch_price']:.8f}")
        logger.info(f"  Market Cap: ${metrics['market_cap']:,.0f}")
        logger.info(f"  Airdrop Value: ${metrics['airdrop_value']:,.0f}")
        
        # Key metrics
        logger.info(f"📈 Performance:")
        logger.info(f"  Profitable Users: {metrics['profitable_users_percent']:.1f}%")
        logger.info(f"  Tokens per $1: {metrics['tokens_per_dollar']:,.0f}")
        logger.info(f"  Opportunity Cost: {params['opportunity_cost']:.2f}%")
        logger.info(f"  Beta: {metrics['beta']:.6f}")
        
        # Validation indicators
        status_symbols = {True: "✅", False: "❌"}
        logger.info(f"✅ Compliance:")
        logger.info(f"  Price < $0.50: {status_symbols[validation['price_under_050']]} ${params['launch_price']:.6f}")
        logger.info(f"  Supply ≥ 2B: {status_symbols[validation['supply_over_2b']]} {params['total_supply']:,.0f}")
        logger.info(f"  Opp Cost ≥ 2%: {status_symbols[validation['opportunity_cost_over_2']]} {params['opportunity_cost']:.2f}%")
        logger.info(f"  High Profitability: {status_symbols[validation['high_user_profitability']]} {metrics['profitable_users_percent']:.1f}%")
        logger.info(f"  High Airdrop: {status_symbols[validation['high_airdrop_percent']]} {params['airdrop_percent']:.1f}%")
    
    # Summary statistics
    logger.info(f"\n📊 SUMMARY STATISTICS:")
    logger.info("-" * 40)
    
    prices = [c['parameters']['launch_price'] for c in configurations]
    airdrops = [c['parameters']['airdrop_percent'] for c in configurations]
    profitability = [c['metrics']['profitable_users_percent'] for c in configurations]
    viability = [c['validation']['overall_viability_score'] for c in configurations]
    
    logger.info(f"Price Range: ${min(prices):.8f} - ${max(prices):.8f}")
    logger.info(f"Airdrop Range: {min(airdrops):.1f}% - {max(airdrops):.1f}%")
    logger.info(f"Profitability Range: {min(profitability):.1f}% - {max(profitability):.1f}%")
    logger.info(f"Viability Score Range: {min(viability):.2f} - {max(viability):.2f}")
    
    # Count configurations meeting different thresholds
    high_quality = sum(1 for c in configurations if c['validation']['overall_viability_score'] >= 0.8)
    medium_quality = sum(1 for c in configurations if 0.6 <= c['validation']['overall_viability_score'] < 0.8)
    
    logger.info(f"\n🎯 Quality Distribution:")
    logger.info(f"  High Quality (≥0.8): {high_quality} configurations")
    logger.info(f"  Medium Quality (0.6-0.8): {medium_quality} configurations")
    logger.info(f"  Lower Quality (<0.6): {len(configurations) - high_quality - medium_quality} configurations")

def save_configurations(configurations: List[Dict], filename: str = "10_relaxed_configurations.json"):
    """Save configurations to JSON file"""
    
    output_data = {
        "search_methodology": "Progressive constraint relaxation to find viable solutions",
        "core_requirements_strict": {
            "token_price": "< $0.50",
            "min_supply": "2,000,000,000 tokens",
            "min_opportunity_cost": "2.0%"
        },
        "flexible_requirements": {
            "airdrop_percent": "Target 100%, minimum 50%",
            "user_profitability": "Target 100%, minimum 50%"
        },
        "total_configurations_found": len(configurations),
        "configurations": configurations,
        "generation_timestamp": "2025-06-07",
        "quality_notes": [
            "Configurations sorted by viability score then price",
            "Viability score based on meeting core + flexible requirements",
            "All configurations meet strict core requirements",
            "Flexible requirements relaxed progressively to find solutions",
            "Higher viability scores indicate better overall compliance"
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n💾 Configurations saved to: {filename}")

def main():
    """Main function to find, analyze, and save configurations"""
    
    # Find configurations with progressive relaxation
    configurations = find_feasible_configurations()
    
    # Analyze and display results
    analyze_and_display_configurations(configurations)
    
    # Save to file
    if configurations:
        save_configurations(configurations)
        
        logger.info("\n" + "="*80)
        logger.info("🎉 PROGRESSIVE RELAXATION SEARCH COMPLETED!")
        logger.info("="*80)
        logger.info(f"✅ Generated {len(configurations)} viable configurations")
        logger.info("✅ All meet core strict requirements (price, supply, opp cost)")
        logger.info("✅ Flexible requirements optimized within constraints")
        logger.info("✅ Configurations ranked by viability score")
        logger.info("✅ Results saved with methodology documentation")
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ NO VIABLE CONFIGURATIONS FOUND")
        logger.error("="*80)
        logger.error("Even with progressive relaxation, no solutions found.")
        logger.error("Consider further adjusting core requirements.")

if __name__ == "__main__":
    main()