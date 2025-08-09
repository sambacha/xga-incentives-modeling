#!/usr/bin/env python3
"""
Mass Market Strategy Track Allocation Generator with Options-Based Structure
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

# Options parameters
STRIKE_PRICE_RATIO = 0.50  # 50% of launch price
VESTING_PERIOD = 365        # 1 year vesting

# Load the mass market strategy from points.json
def load_mass_market_strategy() -> Dict:
    """Load the mass market strategy parameters from points.json"""
    with open('points.json', 'r') as f:
        data = json.load(f)
    return data[0]  # First (and only) strategy in the file

@dataclass
class TrackAllocationResult:
    """Enhanced allocation result with options-based metrics"""
    track_type: str
    participant_id: str
    participant_tier: str  # retail, power_user, whale, institution
    capital_invested: float
    points_earned: float
    capital_equivalent: float
    risk_factor: float
    option_quantity: float          # Number of options granted
    strike_price: float             # Per-token strike price
    intrinsic_value: float          # (Launch price - Strike price) * Quantity
    time_value: float               # Additional value from vesting period
    total_option_value: float       # Intrinsic + Time value
    roi_percent: float
    model_used: str
    profitability_tier: str
    accessibility_score: float

class OptionsBasedAllocationModel:
    """Options-based allocation models"""
    
    def __init__(self, strategy_params: Dict):
        self.params = strategy_params["parameters"]
        self.launch_price = self.params["launch_price"]
        self.strike_price = self.launch_price * STRIKE_PRICE_RATIO
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
    
    def calculate_options_allocation(self, capital: float, model_type: str) -> float:
        """Calculate option quantity using mass market optimized models"""
        tier = self.get_participant_tier(capital)
        segment_name = self.get_segment_name(tier)
        segment_data = self.segment_data[segment_name]
        
        # Base allocation calculation
        if model_type == "linear":
            option_qty = self._linear_allocation(capital)
        elif model_type == "quadratic":
            option_qty = self._quadratic_allocation(capital)
        elif model_type == "tiered":
            option_qty = self._tiered_allocation(capital)
        else:
            option_qty = self._quadratic_allocation(capital)  # Default
        
        # Apply mass market enhancement based on segment performance
        base_roi = segment_data["roi_percent"]
        enhancement_factor = min(2.0, np.log10(base_roi / 100 + 1))  # Logarithmic enhancement
        
        return option_qty * enhancement_factor
    
    def calculate_option_value(self, option_quantity: float) -> Tuple[float, float, float]:
        """Calculate option value components"""
        # Intrinsic value: (Launch price - Strike price) * Quantity
        intrinsic = (self.launch_price - self.strike_price) * option_quantity
        
        # Time value: Based on Black-Scholes approximation
        # Simplified: 10% of intrinsic value per year of vesting
        time_value = intrinsic * (VESTING_PERIOD / 365) * 0.10
        
        return intrinsic, time_value, intrinsic + time_value
    
    def get_profitability_tier(self, roi_percent: float) -> str:
        """Determine profitability tier based on ROI"""
        if roi_percent > 500:
            return "Extremely High"
        elif roi_percent > 200:
            return "Very High"
        elif roi_percent > 100:
            return "High"
        elif roi_percent > 50:
            return "Moderate"
        else:
            return "Low"
    
    def calculate_roi(self, total_value: float, capital_invested: float) -> float:
        """Calculate ROI: (total value - capital) / capital * 100"""
        if capital_invested <= 0:
            return 0
        return ((total_value - capital_invested) / capital_invested) * 100
    
    def _linear_allocation(self, capital: float) -> float:
        """Linear allocation model"""
        allocation_ratio = capital / 100_000_000  # Reduced denominator
        return self.total_airdrop * allocation_ratio
    
    def _quadratic_allocation(self, capital: float) -> float:
        """Quadratic allocation model with mass market boost"""
        sqrt_capital = np.sqrt(min(capital, 10_000_000))  # Higher cap
        allocation_ratio = sqrt_capital / 1_000_000       # Reduced ratio
        base_allocation = self.total_airdrop * allocation_ratio
        
        # Mass market boost for smaller participants
        if capital < 10000:
            return base_allocation * 1.2  # Reduced boost
        return base_allocation
    
    def _tiered_allocation(self, capital: float) -> float:
        """Enhanced tiered allocation for mass market"""
        # Conservative multipliers
        if capital < 1000:
            multiplier = 1.5
        elif capital < 10000:
            multiplier = 1.2
        elif capital < 100000:
            multiplier = 1.0
        else:
            multiplier = 0.8
        
        base_allocation = (capital / 10_000_000) * self.total_airdrop
        return base_allocation * multiplier
    
    def calculate_accessibility_score(self, capital: float, total_value: float) -> float:
        """Calculate accessibility score (0-100) based on mass market principles"""
        # Price accessibility (40% weight)
        price_score_val = (0.5 - self.launch_price) / 0.4 * 100
        price_score = max(0, min(100, price_score_val))  # Clamped between 0-100
        
        # Capital barrier score (30% weight)
        barrier_score = max(0, 100 - (max(capital, 100) / 100))  # Cap at $10,000
        
        # ROI score (30% weight) with logarithmic scaling
        roi = self.calculate_roi(total_value, capital)
        if roi <= 0:
            roi_score = 0
        else:
            roi_score = min(100, 15 * np.log10(roi + 1))  # Reduced scaling
        
        return (price_score * 0.4 + barrier_score * 0.3 + roi_score * 0.3)

class OptionsTrackCalculator:
    """Options-based track calculator"""
    
    def __init__(self, strategy_params: Dict):
        self.allocation_model = OptionsBasedAllocationModel(strategy_params)
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
            performance_factor = 0.7 + (participant["performance"] * participant["uptime"] / 100)
            
            # Bond face value is validator count * base value
            face_value = validators * base_validator_value
            
            # Bond equivalent value with time and performance adjustment
            capital_equivalent = face_value * duration_multiplier * performance_factor
            capital_invested = face_value  # Original validator value
            
            # Get option quantity using quadratic model
            option_quantity = self.allocation_model.calculate_options_allocation(
                capital_equivalent, "quadratic"
            )
            
            # Calculate option value components
            intrinsic, time_value, total_value = self.allocation_model.calculate_option_value(option_quantity)
            
            # Calculate ROI and profitability
            roi_percent = self.allocation_model.calculate_roi(total_value, capital_invested)
            profitability = self.allocation_model.get_profitability_tier(roi_percent)
            
            # Determine tier and accessibility
            tier = self.allocation_model.get_participant_tier(capital_invested)
            accessibility = self.allocation_model.calculate_accessibility_score(
                capital_invested, total_value
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
                option_quantity=option_quantity,
                strike_price=self.allocation_model.strike_price,
                intrinsic_value=intrinsic,
                time_value=time_value,
                total_option_value=total_value,
                roi_percent=roi_percent,
                model_used="quadratic_enhanced",
                profitability_tier=profitability,
                accessibility_score=accessibility
            )
            results.append(result)
        
        return results
    
    # Other track calculations follow similar pattern (not shown for brevity)
    # [Implementation identical to previous but using options model]

class OptionsAnalysisGenerator:
    """Generate comprehensive analysis for options-based strategy"""
    
    def __init__(self, strategy_data: Dict, utilization_factor: float = 1.0):
        self.strategy_data = strategy_data
        self.calculator = OptionsTrackCalculator(strategy_data)
        self.utilization_factor = utilization_factor
    
    def generate_all_allocations(self) -> List[TrackAllocationResult]:
        """Generate allocations for all tracks"""
        all_results = []
        
        print("Calculating Node Operator allocations...")
        all_results.extend(self.calculator.calculate_node_operator_allocations())
        
        # Add other tracks here (Risk Underwriter, Liquidity Provider, etc.)
        
        # Calculate available airdrop based on utilization factor
        available = self.calculator.allocation_model.total_airdrop * self.utilization_factor
        
        # Validate total allocation doesn't exceed available supply
        total_options = sum(r.option_quantity for r in all_results)
        
        if total_options > available:
            print(f"WARNING: Over-allocated {total_options:,.0f} options vs {available:,.0f} available. Scaling down.")
            scale_factor = available / total_options
            for r in all_results:
                r.option_quantity *= scale_factor
                # Recalculate option values
                intrinsic, time_value, total_value = self.calculator.allocation_model.calculate_option_value(r.option_quantity)
                r.intrinsic_value = intrinsic
                r.time_value = time_value
                r.total_option_value = total_value
                # Recalculate metrics
                r.roi_percent = self.calculator.allocation_model.calculate_roi(
                    r.total_option_value, r.capital_invested
                )
                r.profitability_tier = self.calculator.allocation_model.get_profitability_tier(r.roi_percent)
                r.accessibility_score = self.calculator.allocation_model.calculate_accessibility_score(
                    r.capital_invested, r.total_option_value
                )
        elif total_options < available:
            print(f"NOTE: Under-allocated {available - total_options:,.0f} options (utilization: {self.utilization_factor})")
        
        return all_results
    
    # Analysis and visualization methods would be updated for options metrics
    # [Implementation similar to previous but with option-focused metrics]

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Options-Based Allocation Generator')
    parser.add_argument('--utilization', type=float, default=1.0,
                        help='Airdrop utilization factor (0.0-1.0)')
    args = parser.parse_args()
    
    # Load strategy, initialize, generate allocations, etc.
    # [Similar structure to previous implementation]

if __name__ == "__main__":
    main()
