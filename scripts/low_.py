#!/usr/bin/env python3
"""
Low Price Track-Specific Allocation Generator

This script implements the track-specific allocation models for the low price scenario
as documented in LOW_PRICE_OPTIMIZATION_ANALYSIS.md. It generates allocation results
for Node Operator, Risk Underwriter, Liquidity Provider, and Auction Participant tracks.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

# Low price scenario parameters from the analysis
LOW_PRICE_SCENARIO = {
    "launch_price": 0.10,
    "total_supply": 1_350_000_000,  # Using simple_low_price_solution.json
    "airdrop_percent": 20.0,
    "market_cap": 135_000_000,
    "opportunity_cost": 0.5,
    "volatility": 20.0,
    "revenue_share": 40.0
}

@dataclass
class AllocationResult:
    """Result of allocation calculation for a participant"""
    track_type: str
    participant_id: str
    capital_invested: float
    points_earned: float
    capital_equivalent: float
    risk_factor: float
    allocation_tokens: float
    allocation_value: float
    model_used: str

class AllocationModel:
    """Implementation of the four allocation models from the analysis"""
    
    @staticmethod
    def linear_model(capital: float, total_airdrop: float) -> float:
        """Linear allocation model used for Risk Underwriters"""
        allocation_ratio = capital / 1_000_000
        return total_airdrop * allocation_ratio * 0.01
    
    @staticmethod
    def quadratic_model(capital: float, total_airdrop: float) -> float:
        """Quadratic allocation model used for Node Operators and LPs"""
        sqrt_capital = np.sqrt(min(capital, 1_000_000))
        allocation_ratio = sqrt_capital / 10_000
        return total_airdrop * allocation_ratio * 0.1
    
    @staticmethod
    def logarithmic_model(capital: float, total_airdrop: float) -> float:
        """Logarithmic allocation model for maximum whale protection"""
        log_capital = np.log10(max(capital, 100))
        allocation_ratio = log_capital / 6
        return total_airdrop * allocation_ratio * 0.05
    
    @staticmethod
    def tiered_model(capital: float, total_airdrop: float) -> float:
        """Tiered allocation model used for Auction Participants"""
        # Determine multiplier based on capital tier
        if capital < 1000:
            multiplier = 1.5  # Retail advantage
        elif capital < 10000:
            multiplier = 1.2  # Power users
        elif capital < 100000:
            multiplier = 1.0  # Whales
        else:
            multiplier = 0.8  # Institutions penalized
        
        base_allocation = (capital / 500_000) * total_airdrop * 0.01
        return base_allocation * multiplier

class NodeOperatorTrack:
    """Node Operator track implementation"""
    
    def __init__(self):
        self.base_capital_per_validator = 5000  # Reduced from $10K per analysis
        self.risk_factor = 0.8  # Lower volatility due to infrastructure stability
        self.model = "quadratic"
    
    def calculate_points(self, validators: int, duration_months: int, 
                        performance_score: float = 1.0, uptime: float = 99.0) -> float:
        """Calculate points: validators × duration × performance × uptime"""
        return validators * duration_months * performance_score * (uptime / 100)
    
    def calculate_capital_equivalent(self, points: float, validators: int) -> float:
        """Capital equivalent: Points × $5,000 ÷ validators_operated"""
        return points * self.base_capital_per_validator / validators
    
    def calculate_allocation(self, capital_equivalent: float, total_airdrop: float) -> float:
        """Use quadratic model for allocation"""
        return AllocationModel.quadratic_model(capital_equivalent, total_airdrop)

class RiskUnderwriterTrack:
    """Risk Underwriter track implementation"""
    
    def __init__(self):
        self.token_prices = {"FOLD": 1.0, "EIGEN": 3.0}
        self.risk_factor = 1.2  # Higher volatility due to insurance risk
        self.model = "linear"
    
    def calculate_points(self, tokens_staked: float, duration_months: int, 
                        token_type: str = "FOLD") -> float:
        """Calculate points with duration multiplier"""
        duration_multiplier = self.get_duration_multiplier(duration_months)
        base_points = tokens_staked * (duration_months / 12)
        return base_points * duration_multiplier
    
    def get_duration_multiplier(self, duration_months: int) -> float:
        """Duration multipliers: 1.0x (≤6mo) to 2.0x (>24mo)"""
        if duration_months <= 6:
            return 1.0
        elif duration_months <= 12:
            return 1.2
        elif duration_months <= 24:
            return 1.5
        else:
            return 2.0
    
    def calculate_capital_equivalent(self, tokens_staked: float, token_type: str = "FOLD") -> float:
        """Capital equivalent: Direct staking amount × token_price"""
        return tokens_staked * self.token_prices[token_type]
    
    def calculate_allocation(self, capital_equivalent: float, total_airdrop: float) -> float:
        """Use linear model for allocation"""
        return AllocationModel.linear_model(capital_equivalent, total_airdrop)

class LiquidityProviderTrack:
    """Liquidity Provider track implementation"""
    
    def __init__(self):
        self.lst_price = 3000  # Current ETH price
        self.risk_factor = 0.6  # Lower risk for stable pools, 1.0 for standard
        self.model = "quadratic"
    
    def calculate_points(self, lst_amount: float, duration_months: int, 
                        bonus_multiplier: float = 1.0) -> float:
        """Calculate points: LST_amount × (duration/12) × bonus_multiplier"""
        return lst_amount * (duration_months / 12) * bonus_multiplier
    
    def calculate_capital_equivalent(self, lst_amount: float) -> float:
        """Capital equivalent: LST_amount × $3,000"""
        return lst_amount * self.lst_price
    
    def calculate_allocation(self, capital_equivalent: float, total_airdrop: float) -> float:
        """Use quadratic model for allocation"""
        return AllocationModel.quadratic_model(capital_equivalent, total_airdrop)

class AuctionParticipantTrack:
    """Auction Participant track implementation"""
    
    def __init__(self):
        self.risk_factor = 1.0  # Variable based on accuracy
        self.model = "tiered"
    
    def calculate_points(self, total_bids: int, successful_bids: int, 
                        bid_accuracy: float, total_bid_value: float) -> float:
        """Calculate points: total_bid_value × success_rate × accuracy"""
        success_rate = successful_bids / total_bids if total_bids > 0 else 0
        return total_bid_value * success_rate * (bid_accuracy / 100)
    
    def calculate_capital_equivalent(self, total_bids: int, successful_bids: int,
                                   bid_accuracy: float, total_bid_value: float) -> float:
        """Capital equivalent: Performance-adjusted bid value"""
        if total_bids > 0:
            avg_bid_value = total_bid_value / total_bids
            performance_factor = (successful_bids / total_bids) * (bid_accuracy / 100)
            return avg_bid_value * (1 + performance_factor)
        return 0
    
    def get_risk_factor(self, bid_accuracy: float) -> float:
        """Variable risk factor based on bid accuracy"""
        if bid_accuracy < 50:
            return 1.5  # Higher risk for poor accuracy
        elif bid_accuracy > 80:
            return 0.8  # Lower risk for high accuracy
        return 1.0
    
    def calculate_allocation(self, capital_equivalent: float, total_airdrop: float) -> float:
        """Use tiered model for allocation"""
        return AllocationModel.tiered_model(capital_equivalent, total_airdrop)

class LowPriceAllocationGenerator:
    """Main generator for low price track allocations"""
    
    def __init__(self, scenario_params: Dict):
        self.scenario = scenario_params
        self.total_airdrop = scenario_params["total_supply"] * (scenario_params["airdrop_percent"] / 100)
        
        # Initialize track calculators
        self.node_operator = NodeOperatorTrack()
        self.risk_underwriter = RiskUnderwriterTrack()
        self.liquidity_provider = LiquidityProviderTrack()
        self.auction_participant = AuctionParticipantTrack()
        
        self.results = []
    
    def generate_sample_participants(self) -> Dict[str, List[Dict]]:
        """Generate sample participants for each track"""
        
        # Node Operator participants (various sizes)
        node_operators = [
            {"id": "NO_Small", "validators": 1, "duration": 6, "performance": 0.95, "uptime": 98.5},
            {"id": "NO_Medium", "validators": 5, "duration": 12, "performance": 1.0, "uptime": 99.2},
            {"id": "NO_Large", "validators": 20, "duration": 18, "performance": 1.05, "uptime": 99.8},
            {"id": "NO_Enterprise", "validators": 50, "duration": 24, "performance": 1.1, "uptime": 99.9}
        ]
        
        # Risk Underwriter participants
        risk_underwriters = [
            {"id": "RU_Retail", "tokens": 1000, "duration": 3, "token_type": "FOLD"},
            {"id": "RU_Power", "tokens": 10000, "duration": 12, "token_type": "FOLD"},
            {"id": "RU_Whale", "tokens": 100000, "duration": 18, "token_type": "EIGEN"},
            {"id": "RU_Institution", "tokens": 500000, "duration": 36, "token_type": "EIGEN"}
        ]
        
        # Liquidity Provider participants
        liquidity_providers = [
            {"id": "LP_Small", "lst_amount": 0.5, "duration": 6, "bonus": 1.0},
            {"id": "LP_Medium", "lst_amount": 5.0, "duration": 12, "bonus": 1.1},
            {"id": "LP_Large", "lst_amount": 50.0, "duration": 18, "bonus": 1.2},
            {"id": "LP_Whale", "lst_amount": 200.0, "duration": 24, "bonus": 1.3}
        ]
        
        # Auction Participant participants
        auction_participants = [
            {"id": "AP_Novice", "total_bids": 10, "successful": 3, "accuracy": 45.0, "bid_value": 5000},
            {"id": "AP_Skilled", "total_bids": 25, "successful": 18, "accuracy": 75.0, "bid_value": 25000},
            {"id": "AP_Expert", "total_bids": 50, "successful": 42, "accuracy": 85.0, "bid_value": 100000},
            {"id": "AP_Professional", "total_bids": 100, "successful": 88, "accuracy": 92.0, "bid_value": 500000}
        ]
        
        return {
            "node_operators": node_operators,
            "risk_underwriters": risk_underwriters,
            "liquidity_providers": liquidity_providers,
            "auction_participants": auction_participants
        }
    
    def calculate_track_allocations(self) -> List[AllocationResult]:
        """Calculate allocations for all tracks"""
        participants = self.generate_sample_participants()
        results = []
        
        # Process Node Operators
        for participant in participants["node_operators"]:
            points = self.node_operator.calculate_points(
                participant["validators"], participant["duration"],
                participant["performance"], participant["uptime"]
            )
            capital_equiv = self.node_operator.calculate_capital_equivalent(
                points, participant["validators"]
            )
            allocation = self.node_operator.calculate_allocation(capital_equiv, self.total_airdrop)
            
            result = AllocationResult(
                track_type="Node Operator",
                participant_id=participant["id"],
                capital_invested=participant["validators"] * self.node_operator.base_capital_per_validator,
                points_earned=points,
                capital_equivalent=capital_equiv,
                risk_factor=self.node_operator.risk_factor,
                allocation_tokens=allocation,
                allocation_value=allocation * self.scenario["launch_price"],
                model_used=self.node_operator.model
            )
            results.append(result)
        
        # Process Risk Underwriters
        for participant in participants["risk_underwriters"]:
            points = self.risk_underwriter.calculate_points(
                participant["tokens"], participant["duration"], participant["token_type"]
            )
            capital_equiv = self.risk_underwriter.calculate_capital_equivalent(
                participant["tokens"], participant["token_type"]
            )
            allocation = self.risk_underwriter.calculate_allocation(capital_equiv, self.total_airdrop)
            
            result = AllocationResult(
                track_type="Risk Underwriter",
                participant_id=participant["id"],
                capital_invested=capital_equiv,
                points_earned=points,
                capital_equivalent=capital_equiv,
                risk_factor=self.risk_underwriter.risk_factor,
                allocation_tokens=allocation,
                allocation_value=allocation * self.scenario["launch_price"],
                model_used=self.risk_underwriter.model
            )
            results.append(result)
        
        # Process Liquidity Providers
        for participant in participants["liquidity_providers"]:
            points = self.liquidity_provider.calculate_points(
                participant["lst_amount"], participant["duration"], participant["bonus"]
            )
            capital_equiv = self.liquidity_provider.calculate_capital_equivalent(
                participant["lst_amount"]
            )
            allocation = self.liquidity_provider.calculate_allocation(capital_equiv, self.total_airdrop)
            
            result = AllocationResult(
                track_type="Liquidity Provider",
                participant_id=participant["id"],
                capital_invested=capital_equiv,
                points_earned=points,
                capital_equivalent=capital_equiv,
                risk_factor=self.liquidity_provider.risk_factor,
                allocation_tokens=allocation,
                allocation_value=allocation * self.scenario["launch_price"],
                model_used=self.liquidity_provider.model
            )
            results.append(result)
        
        # Process Auction Participants
        for participant in participants["auction_participants"]:
            points = self.auction_participant.calculate_points(
                participant["total_bids"], participant["successful"],
                participant["accuracy"], participant["bid_value"]
            )
            capital_equiv = self.auction_participant.calculate_capital_equivalent(
                participant["total_bids"], participant["successful"],
                participant["accuracy"], participant["bid_value"]
            )
            allocation = self.auction_participant.calculate_allocation(capital_equiv, self.total_airdrop)
            
            result = AllocationResult(
                track_type="Auction Participant",
                participant_id=participant["id"],
                capital_invested=participant["bid_value"],
                points_earned=points,
                capital_equivalent=capital_equiv,
                risk_factor=self.auction_participant.get_risk_factor(participant["accuracy"]),
                allocation_tokens=allocation,
                allocation_value=allocation * self.scenario["launch_price"],
                model_used=self.auction_participant.model
            )
            results.append(result)
        
        return results
    
    def generate_analysis_report(self, results: List[AllocationResult]) -> Dict:
        """Generate comprehensive analysis report"""
        df = pd.DataFrame([{
            "Track": r.track_type,
            "Participant": r.participant_id,
            "Capital_Invested": r.capital_invested,
            "Points": r.points_earned,
            "Capital_Equivalent": r.capital_equivalent,
            "Risk_Factor": r.risk_factor,
            "Allocation_Tokens": r.allocation_tokens,
            "Allocation_Value": r.allocation_value,
            "Model": r.model_used,
            "ROI_Percent": (r.allocation_value / r.capital_invested) * 100 if r.capital_invested > 0 else 0
        } for r in results])
        
        # Calculate summary statistics
        summary = {}
        for track in df["Track"].unique():
            track_data = df[df["Track"] == track]
            summary[track] = {
                "total_allocation_tokens": track_data["Allocation_Tokens"].sum(),
                "total_allocation_value": track_data["Allocation_Value"].sum(),
                "avg_roi_percent": track_data["ROI_Percent"].mean(),
                "total_capital_invested": track_data["Capital_Invested"].sum(),
                "allocation_model": track_data["Model"].iloc[0],
                "risk_factor": track_data["Risk_Factor"].mean()
            }
        
        return {
            "scenario_parameters": self.scenario,
            "detailed_results": df.to_dict("records"),
            "track_summary": summary,
            "total_tokens_allocated": df["Allocation_Tokens"].sum(),
            "total_value_allocated": df["Allocation_Value"].sum(),
            "average_roi": df["ROI_Percent"].mean()
        }
    
    def create_visualizations(self, results: List[AllocationResult]) -> None:
        """Create visualization charts"""
        df = pd.DataFrame([{
            "Track": r.track_type,
            "Participant": r.participant_id,
            "Capital_Invested": r.capital_invested,
            "Allocation_Tokens": r.allocation_tokens,
            "Allocation_Value": r.allocation_value,
            "ROI_Percent": (r.allocation_value / r.capital_invested) * 100 if r.capital_invested > 0 else 0
        } for r in results])
        
        # Set up the plotting style with fallbacks
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Low Price Airdrop Track Allocation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Token allocation by track
        track_allocation = df.groupby("Track")["Allocation_Tokens"].sum()
        axes[0, 0].pie(track_allocation.values, labels=track_allocation.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Token Allocation Distribution by Track')
        
        # 2. ROI comparison by track
        df.boxplot(column="ROI_Percent", by="Track", ax=axes[0, 1])
        axes[0, 1].set_title('ROI Distribution by Track')
        axes[0, 1].set_xlabel('Track Type')
        axes[0, 1].set_ylabel('ROI (%)')
        
        # 3. Capital vs Allocation scatter
        colors = {'Node Operator': 'blue', 'Risk Underwriter': 'red', 
                 'Liquidity Provider': 'green', 'Auction Participant': 'orange'}
        for track in df["Track"].unique():
            track_data = df[df["Track"] == track]
            axes[1, 0].scatter(track_data["Capital_Invested"], track_data["Allocation_Tokens"], 
                              label=track, color=colors[track], alpha=0.7)
        axes[1, 0].set_xlabel('Capital Invested ($)')
        axes[1, 0].set_ylabel('Token Allocation')
        axes[1, 0].set_title('Capital vs Token Allocation')
        axes[1, 0].legend()
        axes[1, 0].set_xscale('log')
        
        # 4. Allocation value by participant
        df_sorted = df.sort_values("Allocation_Value")
        axes[1, 1].barh(df_sorted["Participant"], df_sorted["Allocation_Value"])
        axes[1, 1].set_xlabel('Allocation Value ($)')
        axes[1, 1].set_title('Allocation Value by Participant')
        
        plt.tight_layout()
        plt.savefig('low_price_track_allocation_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as: low_price_track_allocation_analysis.png")
        plt.close()  # Close figure to free memory

def main():
    """Main execution function"""
    print("=" * 80)
    print("LOW PRICE TRACK-SPECIFIC ALLOCATION GENERATOR")
    print("=" * 80)
    print("Implementing analysis from LOW_PRICE_OPTIMIZATION_ANALYSIS.md")
    print(f"Scenario: ${LOW_PRICE_SCENARIO['launch_price']} launch price")
    print(f"Total Supply: {LOW_PRICE_SCENARIO['total_supply']:,} tokens")
    print(f"Airdrop Pool: {LOW_PRICE_SCENARIO['total_supply'] * (LOW_PRICE_SCENARIO['airdrop_percent']/100):,.0f} tokens")
    print("-" * 80)
    
    # Initialize generator
    generator = LowPriceAllocationGenerator(LOW_PRICE_SCENARIO)
    
    # Calculate allocations
    print("Calculating track-specific allocations...")
    results = generator.calculate_track_allocations()
    
    # Generate analysis report
    print("Generating analysis report...")
    analysis = generator.generate_analysis_report(results)
    
    # Save results to JSON
    output_file = "low_price_track_allocations_results.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("ALLOCATION SUMMARY")
    print("=" * 80)
    
    for track, summary in analysis["track_summary"].items():
        print(f"\n{track}:")
        print(f"  Model: {summary['allocation_model']}")
        print(f"  Total Tokens: {summary['total_allocation_tokens']:,.0f}")
        print(f"  Total Value: ${summary['total_allocation_value']:,.2f}")
        print(f"  Average ROI: {summary['avg_roi_percent']:.1f}%")
        print(f"  Risk Factor: {summary['risk_factor']:.1f}x")
    
    print(f"\nOverall Statistics:")
    print(f"  Total Tokens Allocated: {analysis['total_tokens_allocated']:,.0f}")
    print(f"  Total Value Allocated: ${analysis['total_value_allocated']:,.2f}")
    print(f"  Average ROI: {analysis['average_roi']:.1f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    try:
        generator.create_visualizations(results)
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
        print("Analysis data has been saved successfully to JSON file.")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Generated files:")
    print(f"  - {output_file}")
    print("  - low_price_track_allocation_analysis.png")

if __name__ == "__main__":
    main()