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
    
    def calculate_risk_underwriter_allocations(self) -> List[TrackAllocationResult]:
        """Calculate Risk Underwriter allocations"""
        participants = [
            {"id": "RU_Micro", "tokens": 500, "duration": 3, "token_type": "FOLD"},
            {"id": "RU_Retail", "tokens": 2000, "duration": 6, "token_type": "FOLD"},
            {"id": "RU_Power", "tokens": 15000, "duration": 12, "token_type": "FOLD"},
            {"id": "RU_Whale", "tokens": 75000, "duration": 18, "token_type": "EIGEN"},
            {"id": "RU_Institution", "tokens": 300000, "duration": 24, "token_type": "EIGEN"}
        ]
        
        results = []
        token_prices = {"FOLD": 1.0, "EIGEN": 3.0}
        
        for participant in participants:
            # Duration multiplier
            duration = participant["duration"]
            if duration <= 6:
                multiplier = 1.0
            elif duration <= 12:
                multiplier = 1.3  # Increased from 1.2
            elif duration <= 18:
                multiplier = 1.7  # Increased from 1.5
            else:
                multiplier = 2.2  # Increased from 2.0
            
            # Calculate points and capital
            base_points = participant["tokens"] * (duration / 12)
            points = base_points * multiplier
            capital_invested = participant["tokens"] * token_prices[participant["token_type"]]
            
            # Get allocation using linear model
            allocation = self.allocation_model.calculate_mass_market_allocation(
                capital_invested, "linear"
            )
            allocation_value = allocation * self.allocation_model.launch_price
            
            # Calculate ROI and profitability
            roi_percent = self.allocation_model.calculate_roi(allocation_value, capital_invested)
            profitability = self.allocation_model.get_profitability_tier(roi_percent)
            
            tier = self.allocation_model.get_participant_tier(capital_invested)
            accessibility = self.allocation_model.calculate_accessibility_score(
                capital_invested, allocation_value
            )
            
            result = TrackAllocationResult(
                track_type="Risk Underwriter",
                participant_id=participant["id"],
                participant_tier=tier,
                capital_invested=capital_invested,
                points_earned=points,
                capital_equivalent=capital_invested,
                risk_factor=1.2,
                allocation_tokens=allocation,
                allocation_value=allocation_value,
                roi_percent=roi_percent,
                model_used="linear_enhanced",
                profitability_tier=profitability,
                accessibility_score=accessibility
            )
            results.append(result)
        
        return results
    
    def calculate_liquidity_provider_allocations(self) -> List[TrackAllocationResult]:
        """Calculate Liquidity Provider allocations"""
        participants = [
            {"id": "LP_Micro", "lst_amount": 0.2, "duration": 3, "bonus": 1.0},
            {"id": "LP_Small", "lst_amount": 1.0, "duration": 6, "bonus": 1.1},
            {"id": "LP_Medium", "lst_amount": 5.0, "duration": 12, "bonus": 1.2},
            {"id": "LP_Large", "lst_amount": 25.0, "duration": 18, "bonus": 1.3},
            {"id": "LP_Whale", "lst_amount": 100.0, "duration": 24, "bonus": 1.4}
        ]
        
        results = []
        lst_price = 3000  # ETH price
        
        for participant in participants:
            # Calculate points and capital
            points = (participant["lst_amount"] * (participant["duration"] / 12) * 
                     participant["bonus"])
            capital_invested = participant["lst_amount"] * lst_price
            
            # Get allocation using quadratic model
            allocation = self.allocation_model.calculate_mass_market_allocation(
                capital_invested, "quadratic"
            )
            allocation_value = allocation * self.allocation_model.launch_price
            
            # Calculate ROI and profitability
            roi_percent = self.allocation_model.calculate_roi(allocation_value, capital_invested)
            profitability = self.allocation_model.get_profitability_tier(roi_percent)
            
            tier = self.allocation_model.get_participant_tier(capital_invested)
            accessibility = self.allocation_model.calculate_accessibility_score(
                capital_invested, allocation_value
            )
            
            # Risk factor varies by pool type
            risk_factor = 0.6 if "stable" in participant.get("pool_type", "").lower() else 1.0
            
            result = TrackAllocationResult(
                track_type="Liquidity Provider",
                participant_id=participant["id"],
                participant_tier=tier,
                capital_invested=capital_invested,
                points_earned=points,
                capital_equivalent=capital_invested,
                risk_factor=risk_factor,
                allocation_tokens=allocation,
                allocation_value=allocation_value,
                roi_percent=roi_percent,
                model_used="quadratic_enhanced",
                profitability_tier=profitability,
                accessibility_score=accessibility
            )
            results.append(result)
        
        return results
    
    def calculate_auction_participant_allocations(self) -> List[TrackAllocationResult]:
        """Calculate Auction Participant allocations"""
        participants = [
            {"id": "AP_Novice", "total_bids": 5, "successful": 1, "accuracy": 30.0, "bid_value": 2000},
            {"id": "AP_Learning", "total_bids": 15, "successful": 6, "accuracy": 50.0, "bid_value": 8000},
            {"id": "AP_Skilled", "total_bids": 30, "successful": 20, "accuracy": 75.0, "bid_value": 25000},
            {"id": "AP_Expert", "total_bids": 50, "successful": 42, "accuracy": 85.0, "bid_value": 75000},
            {"id": "AP_Professional", "total_bids": 100, "successful": 88, "accuracy": 92.0, "bid_value": 300000}
        ]
        
        results = []
        
        for participant in participants:
            # Calculate points and performance metrics
            success_rate = participant["successful"] / participant["total_bids"] if participant["total_bids"] > 0 else 0
            points = participant["bid_value"] * success_rate * (participant["accuracy"] / 100)
            
            # Capital equivalent with performance adjustment
            avg_bid_value = participant["bid_value"] / participant["total_bids"]
            performance_factor = success_rate * (participant["accuracy"] / 100)
            capital_equivalent = avg_bid_value * (1 + performance_factor)
            
            # Get allocation using tiered model
            allocation = self.allocation_model.calculate_mass_market_allocation(
                capital_equivalent, "tiered"
            )
            allocation_value = allocation * self.allocation_model.launch_price
            
            # Calculate ROI and profitability
            roi_percent = self.allocation_model.calculate_roi(allocation_value, participant["bid_value"])
            profitability = self.allocation_model.get_profitability_tier(roi_percent)
            
            tier = self.allocation_model.get_participant_tier(participant["bid_value"])
            accessibility = self.allocation_model.calculate_accessibility_score(
                participant["bid_value"], allocation_value
            )
            
            # Variable risk factor based on accuracy
            if participant["accuracy"] < 50:
                risk_factor = 1.5
            elif participant["accuracy"] > 80:
                risk_factor = 0.8
            else:
                risk_factor = 1.0
            
            result = TrackAllocationResult(
                track_type="Auction Participant",
                participant_id=participant["id"],
                participant_tier=tier,
                capital_invested=participant["bid_value"],
                points_earned=points,
                capital_equivalent=capital_equivalent,
                risk_factor=risk_factor,
                allocation_tokens=allocation,
                allocation_value=allocation_value,
                roi_percent=roi_percent,
                model_used="tiered_enhanced",
                profitability_tier=profitability,
                accessibility_score=accessibility
            )
            results.append(result)
        
        return results

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
    
    def create_comprehensive_analysis(self, results: List[TrackAllocationResult]) -> Dict:
        """Create comprehensive analysis report"""
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Track-level analysis
        track_analysis = {}
        for track in df["track_type"].unique():
            track_data = df[df["track_type"] == track]
            track_analysis[track] = {
                "total_participants": len(track_data),
                "total_allocation_tokens": track_data["allocation_tokens"].sum(),
                "total_allocation_value": track_data["allocation_value"].sum(),
                "avg_roi_percent": track_data["roi_percent"].mean(),
                "total_capital_invested": track_data["capital_invested"].sum(),
                "avg_accessibility_score": track_data["accessibility_score"].mean(),
                "profitability_distribution": track_data["profitability_tier"].value_counts().to_dict(),
                "tier_distribution": track_data["participant_tier"].value_counts().to_dict()
            }
        
        # Tier-level analysis
        tier_analysis = {}
        expected_tiers = ['retail', 'power_user', 'whale', 'institution']
        for tier in expected_tiers:
            if tier in df['participant_tier'].unique():
                tier_data = df[df["participant_tier"] == tier]
                tier_analysis[tier] = {
                    "participant_count": len(tier_data),
                    "avg_roi_percent": tier_data["roi_percent"].mean(),
                    "avg_accessibility_score": tier_data["accessibility_score"].mean(),
                    "total_allocation_value": tier_data["allocation_value"].sum(),
                    "capital_efficiency": tier_data["allocation_value"].sum() / tier_data["capital_invested"].sum()
                }
            else:
                tier_analysis[tier] = "No participants in this tier"
        
        # Overall metrics
        total_allocated = df["allocation_tokens"].sum()
        total_available = self.calculator.allocation_model.total_airdrop
        
        overall_metrics = {
            "total_participants": len(df),
            "total_tokens_allocated": total_allocated,
            "total_value_allocated": df["allocation_value"].sum(),
            "average_roi": df["roi_percent"].mean(),
            "average_accessibility": df["accessibility_score"].mean(),
            "total_capital_invested": df["capital_invested"].sum(),
            "capital_efficiency_ratio": df["allocation_value"].sum() / df["capital_invested"].sum(),
            "airdrop_utilization": total_allocated / total_available,
            "mass_market_score": self._calculate_mass_market_score(df)
        }
        
        return {
            "strategy_parameters": self.strategy_data,
            "allocation_parameters": {
                "utilization_factor": self.utilization_factor,
                "total_airdrop_available": total_available
            },
            "detailed_results": df.to_dict("records"),
            "track_analysis": track_analysis,
            "tier_analysis": tier_analysis,
            "overall_metrics": overall_metrics,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_mass_market_score(self, df: pd.DataFrame) -> float:
        """Calculate overall mass market effectiveness score (0-100)"""
        # Accessibility component (40%)
        accessibility_score = df["accessibility_score"].mean() * 0.4
        
        # Retail participation component (30%)
        retail_count = len(df[df["participant_tier"] == "retail"])
        retail_participation = min(100, (retail_count / len(df)) * 200) * 0.3
        
        # ROI distribution component (30%)
        high_roi_count = len(df[df["roi_percent"] > 1000])
        roi_distribution = min(100, (high_roi_count / len(df)) * 100) * 0.3
        
        return accessibility_score + retail_participation + roi_distribution
    
    def create_visualizations(self, results: List[TrackAllocationResult]) -> None:
        """Create comprehensive visualizations"""
        if not results:
            print("Warning: No results to visualize")
            return
            
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Set up plotting with fallback styles
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f'Mass Market Strategy Track Allocation Analysis\n$0.291 Launch Price | {self.utilization_factor*100:.1f}% Airdrop Utilization | 12.4B Supply', 
                    fontsize=16, fontweight='bold')
        
        # Create copies for safe manipulation
        df_plot = df.copy()
        
        # Handle log scale edge cases
        df_plot['roi_percent'] = df_plot['roi_percent'].replace(0, 0.001)
        df_plot['capital_invested'] = df_plot['capital_invested'].clip(lower=1)
        df_plot['allocation_tokens'] = df_plot['allocation_tokens'].clip(lower=1)
        
        # Define consistent color scheme
        track_colors = {
            'Node Operator': '#FF6B6B',
            'Risk Underwriter': '#4ECDC4',
            'Liquidity Provider': '#45B7D1',
            'Auction Participant': '#96CEB4'
        }
        
        # 1. Allocation distribution by track
        track_allocation = df.groupby("track_type")["allocation_tokens"].sum()
        if not track_allocation.empty:
            colors = [track_colors.get(track, '#999999') for track in track_allocation.index]
            axes[0, 0].pie(track_allocation.values, labels=track_allocation.index, 
                          autopct='%1.1f%%', colors=colors)
            axes[0, 0].set_title('Token Allocation Distribution by Track')
        else:
            axes[0, 0].text(0.5, 0.5, 'No allocation data', ha='center', va='center', 
                           transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Token Allocation Distribution by Track')
        
        # 2. ROI by participant tier
        tier_order = ['retail', 'power_user', 'whale', 'institution']
        available_tiers = [t for t in tier_order if t in df_plot['participant_tier'].values]
        
        if available_tiers:
            sns.boxplot(data=df_plot, x='participant_tier', y='roi_percent', 
                       order=available_tiers, ax=axes[0, 1])
            axes[0, 1].set_title('ROI Distribution by Participant Tier')
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_ylabel('ROI (%) - Log Scale')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No participant tier data', ha='center', va='center', 
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('ROI Distribution by Participant Tier')
        
        # 3. Accessibility scores by track
        if not df_plot.empty:
            sns.barplot(data=df_plot, x='track_type', y='accessibility_score', 
                       ax=axes[1, 0], palette=track_colors)
            axes[1, 0].set_title('Average Accessibility Score by Track')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylim(0, 100)
        else:
            axes[1, 0].text(0.5, 0.5, 'No accessibility data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Average Accessibility Score by Track')
        
        # 4. Capital vs Allocation (log scale)
        if not df_plot.empty:
            for track in df_plot["track_type"].unique():
                track_data = df_plot[df_plot["track_type"] == track]
                color = track_colors.get(track, '#999999')
                axes[1, 1].scatter(track_data["capital_invested"], track_data["allocation_tokens"], 
                                 label=track, color=color, alpha=0.7, s=100)
            axes[1, 1].set_xlabel('Capital Invested ($)')
            axes[1, 1].set_ylabel('Token Allocation')
            axes[1, 1].set_title('Capital vs Token Allocation (Log Scale)')
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].set_xlim(left=1)
            axes[1, 1].set_ylim(bottom=1)
        else:
            axes[1, 1].text(0.5, 0.5, 'No capital/allocation data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Capital vs Token Allocation')
        
        # 5. Profitability tier distribution
        if 'profitability_tier' in df_plot and not df_plot['profitability_tier'].empty:
            profitability_counts = df_plot['profitability_tier'].value_counts()
            axes[2, 0].bar(profitability_counts.index, profitability_counts.values, color='#FFB347')
            axes[2, 0].set_title('Profitability Tier Distribution')
            axes[2, 0].tick_params(axis='x', rotation=45)
        else:
            axes[2, 0].text(0.5, 0.5, 'No profitability data', ha='center', va='center', 
                           transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Profitability Tier Distribution')
        
        # 6. Allocation value by participant
        if len(df_plot) > 0:
            df_top = df_plot.nlargest(min(15, len(df_plot)), 'allocation_value')
            if not df_top.empty:
                # Sort for horizontal bar plot
                df_top = df_top.sort_values('allocation_value', ascending=True)
                bars = axes[2, 1].barh(df_top["participant_id"], df_top["allocation_value"])
                axes[2, 1].set_xlabel('Allocation Value ($)')
                axes[2, 1].set_title(f'Top {len(df_top)} Allocation Values by Participant')
                
                # Color bars by track type
                for i, bar in enumerate(bars):
                    track = df_top.iloc[i]["track_type"]
                    color = track_colors.get(track, '#999999')
                    bar.set_color(color)
            else:
                axes[2, 1].text(0.5, 0.5, 'No allocation data', ha='center', va='center', 
                               transform=axes[2, 1].transAxes)
                axes[2, 1].set_title('Allocation Values by Participant')
        else:
            axes[2, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Allocation Values by Participant')
        
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig('mass_market_strategy_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as: mass_market_strategy_analysis.png")
            plt.close(fig)  # Close figure to free memory
        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")
            plt.close(fig)

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
    
    # Create mapping from participant ID to duration
    duration_map = {
        "NO_Micro": 3,
        "NO_Small": 6,
        "NO_Medium": 12,
        "NO_Large": 18,
        "NO_Enterprise": 24
    }
    
    for result in [r for r in results if r.track_type == "Node Operator"]:
        duration = duration_map.get(result.participant_id, 0)
        multiplier = BOND_DURATION_MULTIPLIERS.get(duration, 1.0)
        
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
