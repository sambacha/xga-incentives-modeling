#!/usr/bin/env python3
"""
Mass Market Strategy Track Allocation Generator with Bond-Like Node Operator Commitments
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Bond duration multipliers (3-month increments)
BOND_DURATION_MULTIPLIERS = {
    3: 1.0,    # Base
    6: 1.5,    # +50%
    12: 2.0,   # +100%
    18: 2.5,   # +150%
    24: 3.0    # +200%
}

# Load the mass market strategy from points.json
def load_mass_market_strategy() -> Dict:
    """Load the mass market strategy parameters from points.json"""
    with open('points.json', 'r') as f:
        data = json.load(f)
    return data[0]  # First (and only) strategy in the file

@dataclass
class TrackAllocationResult:
    """Enhanced allocation result with mass market strategy metrics"""
    track_type: str
    participant_id: str
    participant_tier: str  # retail, power_user, whale, institution
    capital_invested: float
    points_earned: float
    capital_equivalent: float
    risk_factor: float
    allocation_tokens: float
    allocation_value: float
    roi_percent: float
    model_used: str
    profitability_tier: str
    accessibility_score: float

class MassMarketAllocationModel:
    """Enhanced allocation models optimized for mass market strategy"""
    
    def __init__(self, strategy_params: Dict):
        self.params = strategy_params["parameters"]
        self.launch_price = self.params["launch_price"]
        self.total_supply = self.params["total_supply"]
        self.airdrop_percent = self.params["airdrop_percent"]
        self.total_airdrop = self.total_supply * (self.airdrop_percent / 100)
        
        # Segment data from points.json
        self.segment_data = {
            "Retail (<$1k)": strategy_params["segment_results"][0],
            "Power Users ($1-10k)": strategy_params["segment_results"][1], 
            "Whales ($10-100k)": strategy_params["segment_results"][2],
            "Institutions (>$100k)": strategy_params["segment_results"][3]
        }
    
    def get_participant_tier(self, capital: float) -> str:
        """Determine participant tier based on capital"""
        if capital < 1000:
            return "retail"
        elif capital < 10000:
            return "power_user"
        elif capital < 100000:
            return "whale"
        else:
            return "institution"
    
    def get_segment_name(self, tier: str) -> str:
        """Get segment name from tier"""
        tier_map = {
            "retail": "Retail (<$1k)",
            "power_user": "Power Users ($1-10k)",
            "whale": "Whales ($10-100k)",
            "institution": "Institutions (>$100k)"
        }
        return tier_map[tier]
    
    def calculate_mass_market_allocation(self, capital: float, model_type: str) -> float:
        """Calculate allocation using mass market optimized models"""
        tier = self.get_participant_tier(capital)
        segment_name = self.get_segment_name(tier)
        segment_data = self.segment_data[segment_name]
        
        # Base allocation calculation
        if model_type == "linear":
            allocation = self._linear_allocation(capital)
        elif model_type == "quadratic":
            allocation = self._quadratic_allocation(capital)
        elif model_type == "tiered":
            allocation = self._tiered_allocation(capital)
        else:
            allocation = self._quadratic_allocation(capital)  # Default
        
        # Apply mass market enhancement based on segment performance
        base_roi = segment_data["roi_percent"]
        enhancement_factor = min(2.0, np.log10(base_roi / 100 + 1))  # Logarithmic enhancement
        
        return allocation * enhancement_factor
    
    def get_profitability_tier(self, roi_percent: float) -> str:
        """Determine profitability tier based on ROI"""
        if roi_percent > 1000:
            return "Extremely High"
        elif roi_percent > 500:
            return "Very High"
        elif roi_percent > 100:
            return "High"
        elif roi_percent > 20:
            return "Moderate"
        else:
            return "Low"
    
    def calculate_roi(self, allocation_value: float, capital_invested: float) -> float:
        """Calculate proper ROI: (gain - cost) / cost * 100"""
        if capital_invested <= 0:
            return 0
        return ((allocation_value - capital_invested) / capital_invested) * 100
    
    def _linear_allocation(self, capital: float) -> float:
        """Linear allocation model"""
        allocation_ratio = capital / 1_000_000
        return self.total_airdrop * allocation_ratio * 0.01
    
    def _quadratic_allocation(self, capital: float) -> float:
        """Quadratic allocation model with mass market boost"""
        sqrt_capital = np.sqrt(min(capital, 1_000_000))
        allocation_ratio = sqrt_capital / 10_000
        base_allocation = self.total_airdrop * allocation_ratio * 0.1
        
        # Mass market boost for smaller participants
        if capital < 10000:
            boost_factor = 1.5  # 50% boost for smaller participants
            return base_allocation * boost_factor
        return base_allocation
    
    def _tiered_allocation(self, capital: float) -> float:
        """Enhanced tiered allocation for mass market"""
        # Enhanced multipliers for mass market strategy
        if capital < 1000:
            multiplier = 2.0  # Increased from 1.5x
        elif capital < 10000:
            multiplier = 1.5  # Increased from 1.2x
        elif capital < 100000:
            multiplier = 1.0  # Standard
        else:
            multiplier = 0.7  # Reduced from 0.8x for institutions
        
        base_allocation = (capital / 500_000) * self.total_airdrop * 0.01
        return base_allocation * multiplier
    
    def calculate_accessibility_score(self, capital: float, allocation_value: float) -> float:
        """Calculate accessibility score (0-100) based on mass market principles"""
        # Price accessibility (40% weight)
        price_score_val = (0.5 - self.launch_price) / 0.4 * 100
        price_score = max(0, min(100, price_score_val))  # Clamped between 0-100
        
        # Capital barrier score (30% weight)
        barrier_score = max(0, 100 - (max(capital, 100) / 100))  # Cap at $10,000
        
        # ROI score (30% weight) with logarithmic scaling
        roi = self.calculate_roi(allocation_value, capital)
        if roi <= 0:
            roi_score = 0
        else:
            roi_score = min(100, 20 * np.log10(roi + 1))
        
        return (price_score * 0.4 + barrier_score * 0.3 + roi_score * 0.3)

class MassMarketTrackCalculator:
    """Enhanced track calculator using mass market strategy"""
    
    def __init__(self, strategy_params: Dict):
        self.allocation_model = MassMarketAllocationModel(strategy_params)
        self.strategy_params = strategy_params
    
    def calculate_node_operator_allocations(self) -> List[TrackAllocationResult]:
        """Calculate Node Operator allocations with bond-like structure"""
        participants = [
            {"id": "NO_Micro", "validators": 1, "duration": 3, "performance": 0.90, "uptime": 97.0},
            {"id": "NO_Small", "validators": 2, "duration": 6, "performance": 0.95, "uptime": 98.5},
            {"id": "NO_Medium", "validators": 5, "duration": 12, "performance": 1.0, "uptime": 99.2},
            {"id": "NO_Large", "validators": 15, "duration": 18, "performance": 1.05, "uptime": 99.5},
            {"id": "NO_Enterprise", "validators": 50, "duration": 24, "performance": 1.1, "uptime": 99.9}
        ]
        
        results = []
        base_validator_value = 3000  # Value per validator
        
        for participant in participants:
            duration = participant["duration"]
            validators = participant["validators"]
            
            # Calculate bond-like capital equivalent
            duration_multiplier = BOND_DURATION_MULTIPLIERS[duration]
            performance_factor = 0.5 + (participant["performance"] * participant["uptime"] / 100)
            
            # Bond face value is validator count * base value
            face_value = validators * base_validator_value
            
            # Bond equivalent value with time and performance adjustment
            capital_equivalent = face_value * duration_multiplier * performance_factor
            capital_invested = face_value  # Original validator value
            
            # Get allocation using quadratic model
            allocation = self.allocation_model.calculate_mass_market_allocation(
                capital_equivalent, "quadratic"
            )
            allocation_value = allocation * self.allocation_model.launch_price
            
            # Calculate ROI and profitability
            roi_percent = self.allocation_model.calculate_roi(allocation_value, capital_invested)
            profitability = self.allocation_model.get_profitability_tier(roi_percent)
            
            # Determine tier and accessibility
            tier = self.allocation_model.get_participant_tier(capital_invested)
            accessibility = self.allocation_model.calculate_accessibility_score(
                capital_invested, allocation_value
            )
            
            # Points represent bond strength
            points = validators * duration * participant["performance"] * (participant["uptime"] / 100)
            
            result = TrackAllocationResult(
                track_type="Node Operator",
                participant_id=participant["id"],
                participant_tier=tier,
                capital_invested=capital_invested,
                points_earned=points,
                capital_equivalent=capital_equivalent,
                risk_factor=0.8,  # Node operators have lower risk
                allocation_tokens=allocation,
                allocation_value=allocation_value,
                roi_percent=roi_percent,
                model_used="quadratic_enhanced",
                profitability_tier=profitability,
                accessibility_score=accessibility
            )
            results.append(result)
        
        return results
    
    # Other track calculations remain unchanged (Risk Underwriter, Liquidity Provider, Auction Participant)
    # [Identical to previous implementation for other tracks]

class MassMarketAnalysisGenerator:
    """Generate comprehensive analysis for mass market strategy"""
    
    def __init__(self, strategy_data: Dict, utilization_factor: float = 1.0):
        self.strategy_data = strategy_data
        self.calculator = MassMarketTrackCalculator(strategy_data)
        self.utilization_factor = utilization_factor
    
    def generate_all_allocations(self) -> List[TrackAllocationResult]:
        """Generate allocations for all tracks"""
        all_results = []
        
        print("Calculating Node Operator allocations...")
        all_results.extend(self.calculator.calculate_node_operator_allocations())
        
        print("Calculating Risk Underwriter allocations...")
        all_results.extend(self.calculator.calculate_risk_underwriter_allocations())
        
        print("Calculating Liquidity Provider allocations...")
        all_results.extend(self.calculator.calculate_liquidity_provider_allocations())
        
        print("Calculating Auction Participant allocations...")
        all_results.extend(self.calculator.calculate_auction_participant_allocations())
        
        # Calculate available airdrop based on utilization factor
        available = self.calculator.allocation_model.total_airdrop * self.utilization_factor
        
        # Validate total allocation doesn't exceed available supply
        total_allocated = sum(r.allocation_tokens for r in all_results)
        
        if total_allocated > available:
            print(f"WARNING: Overallocated {total_allocated:,.0f} tokens vs {available:,.0f} available. Scaling down.")
            scale_factor = available / total_allocated
            for r in all_results:
                r.allocation_tokens *= scale_factor
                r.allocation_value = r.allocation_tokens * self.calculator.allocation_model.launch_price
                # Recalculate metrics with scaled allocation
                r.roi_percent = self.calculator.allocation_model.calculate_roi(
                    r.allocation_value, r.capital_invested
                )
                r.profitability_tier = self.calculator.allocation_model.get_profitability_tier(r.roi_percent)
                r.accessibility_score = self.calculator.allocation_model.calculate_accessibility_score(
                    r.capital_invested, r.allocation_value
                )
        elif total_allocated < available:
            print(f"NOTE: Underallocated {available - total_allocated:,.0f} tokens (utilization: {self.utilization_factor})")
        
        return all_results
    
    # Analysis and visualization methods remain unchanged
    # [Identical to previous implementation]

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Mass Market Strategy Allocation Generator')
    parser.add_argument('--utilization', type=float, default=1.0,
                        help='Airdrop utilization factor (0.0-1.0) to control total allocation')
    args = parser.parse_args()
    
    print("=" * 90)
    print("MASS MARKET STRATEGY TRACK ALLOCATION GENERATOR")
    print("=" * 90)
    print(f"Using airdrop utilization factor: {args.utilization:.1%}")
    print("Loading optimized parameters from points.json")
    
    # Load strategy data
    try:
        strategy_data = load_mass_market_strategy()
    except FileNotFoundError:
        print("ERROR: points.json not found. Please ensure the file exists in the current directory.")
        return
    except Exception as e:
        print(f"ERROR loading points.json: {e}")
        return
    
    params = strategy_data["parameters"]
    print(f"Launch Price: ${params['launch_price']:.6f}")
    print(f"Total Supply: {params['total_supply']:,.0f} tokens")
    print(f"Airdrop Percentage: {params['airdrop_percent']}%")
    print(f"Total Airdrop: {params['total_supply'] * (params['airdrop_percent']/100):,.0f} tokens")
    print(f"Market Cap: ${params['total_supply'] * params['launch_price']:,.0f}")
    print(f"Quality Score: {strategy_data['analysis']['quality_score']:.1f}/100")
    print("-" * 90)
    
    # Initialize generator with utilization factor
    generator = MassMarketAnalysisGenerator(strategy_data, utilization_factor=args.utilization)
    
    # Generate allocations
    print("Generating track-specific allocations...")
    results = generator.generate_all_allocations()
    
    # Create analysis
    print("Creating comprehensive analysis...")
    analysis = generator.create_comprehensive_analysis(results)
    
    # Save results
    output_file = f"mass_market_allocations_{int(args.utilization*100)}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")
    
    # Display summary
    print("\n" + "=" * 90)
    print("MASS MARKET ALLOCATION SUMMARY")
    print("=" * 90)
    
    overall = analysis["overall_metrics"]
    params = analysis["allocation_parameters"]
    
    print(f"Utilization Factor: {args.utilization:.1%}")
    print(f"Total Participants: {overall['total_participants']}")
    print(f"Total Tokens Allocated: {overall['total_tokens_allocated']:,.0f}")
    print(f"Total Airdrop Available: {params['total_airdrop_available']:,.0f}")
    print(f"Utilization Percentage: {overall['airdrop_utilization']:.1%}")
    print(f"Total Value Allocated: ${overall['total_value_allocated']:,.2f}")
    print(f"Average ROI: {overall['average_roi']:,.1f}%")
    print(f"Average Accessibility Score: {overall['average_accessibility']:.1f}/100")
    print(f"Mass Market Score: {overall['mass_market_score']:.1f}/100")
    print(f"Capital Efficiency: {overall['capital_efficiency_ratio']:.2f}x")
    
    # Additional node operator bond analysis
    print("\nNode Operator Bond Analysis:")
    print("-" * 50)
    for result in [r for r in results if r.track_type == "Node Operator"]:
        duration = int(result.participant_id.split("_")[-1])
        multiplier = BOND_DURATION_MULTIPLIERS[duration]
        print(f"{result.participant_id}:")
        print(f"  Validators: {result.capital_invested / 3000:.0f}")
        print(f"  Duration: {duration} months")
        print(f"  Bond Multiplier: {multiplier}x")
        print(f"  Capital Equivalent: ${result.capital_equivalent:,.0f}")
        print(f"  ROI: {result.roi_percent:,.1f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    try:
        generator.create_visualizations(results)
    except Exception as e:
        print(f"Warning: Visualization generation failed: {str(e)}")
        print("Analysis data has been saved successfully to JSON file.")
    
    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)
    print("Generated files:")
    print(f"  - {output_file}")
    print("  - mass_market_strategy_analysis.png")
    print("\nNode operator commitments are structured as bonds with duration-based multipliers:")
    for dur, mult in BOND_DURATION_MULTIPLIERS.items():
        print(f"  - {dur} months: {mult}x face value")

if __name__ == "__main__":
    main()
