"""
Track-specific calculators for multi-track airdrop system
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .types import (
    TrackType, TrackParameters, NodeOperatorParameters,
    RiskUnderwriterParameters, LiquidityProviderParameters,
    AuctionParticipantParameters, AirdropParameters, UserSegment
)


class BaseTrackCalculator(ABC):
    """Base class for track-specific calculators"""
    
    def __init__(self, track_params: TrackParameters):
        self.track_params = track_params
        self.track_type = track_params.track_type
    
    @abstractmethod
    def calculate_points(self) -> float:
        """Calculate points for this track"""
        pass
    
    @abstractmethod
    def points_to_capital_equivalent(self, points: float) -> float:
        """Convert track points to capital equivalent for allocation calculation"""
        pass
    
    @abstractmethod
    def get_risk_factor(self) -> float:
        """Get risk factor for this track (used in volatility calculations)"""
        pass
    
    def get_allocation_model(self) -> str:
        """Get recommended allocation model for this track"""
        return "quadratic"  # Default to quadratic model


class NodeOperatorCalculator(BaseTrackCalculator):
    """Calculator for Node Operator track"""
    
    def __init__(self, track_params: TrackParameters):
        super().__init__(track_params)
        self.params: NodeOperatorParameters = track_params.node_operator
    
    def calculate_points(self) -> float:
        """Calculate points based on validators and performance"""
        return self.params.calculate_points()
    
    def points_to_capital_equivalent(self, points: float) -> float:
        """Convert points to capital equivalent
        
        Node operators have high capital requirements, so we scale points
        to reflect the infrastructure investment
        """
        # Base capital per validator: $5,000 (reduced for better ROI)
        # Adjusted by performance and duration
        base_capital_per_validator = 5000
        capital_equivalent = points * base_capital_per_validator / self.params.validators_operated
        return min(capital_equivalent, 1000000)  # Cap at $1M
    
    def get_risk_factor(self) -> float:
        """Node operators have moderate risk due to slashing"""
        return 0.8  # 80% of base volatility


class RiskUnderwriterCalculator(BaseTrackCalculator):
    """Calculator for Risk Underwriter track"""
    
    def __init__(self, track_params: TrackParameters):
        super().__init__(track_params)
        self.params: RiskUnderwriterParameters = track_params.risk_underwriter
    
    def calculate_points(self) -> float:
        """Calculate points based on staking amount and duration"""
        return self.params.calculate_points()
    
    def points_to_capital_equivalent(self, points: float) -> float:
        """Convert points to capital equivalent
        
        Risk underwriters directly stake capital, so conversion is more direct
        """
        # Points already incorporate duration multiplier
        # Scale by average token price (assume $1 for simplicity)
        token_price = 1.0
        if self.params.token_type == "EIGEN":
            token_price = 3.0  # EIGEN typically higher value
        
        return self.params.tokens_staked * token_price
    
    def get_risk_factor(self) -> float:
        """Risk underwriters take on insurance risk"""
        return 1.2  # 120% of base volatility
    
    def get_allocation_model(self) -> str:
        """Risk underwriters benefit from linear model due to direct staking"""
        return "linear"


class LiquidityProviderCalculator(BaseTrackCalculator):
    """Calculator for Liquidity Provider track"""
    
    def __init__(self, track_params: TrackParameters):
        super().__init__(track_params)
        self.params: LiquidityProviderParameters = track_params.liquidity_provider
    
    def calculate_points(self) -> float:
        """Calculate points based on LST amount and duration"""
        return self.params.calculate_points()
    
    def points_to_capital_equivalent(self, points: float) -> float:
        """Convert points to capital equivalent
        
        LST providers have capital locked in liquid staking
        """
        # LSTs are typically worth close to underlying asset
        # stETH ≈ ETH, assume $3000 per unit for calculation (updated price)
        lst_price = 3000
        return self.params.lst_amount * lst_price
    
    def get_risk_factor(self) -> float:
        """LP risk depends on pool type"""
        if "stable" in self.params.pool_type.lower():
            return 0.6  # Lower risk for stable pools
        return 1.0  # Standard risk
    
    def get_allocation_model(self) -> str:
        """LPs benefit from quadratic model to prevent concentration"""
        return "quadratic"


class AuctionParticipantCalculator(BaseTrackCalculator):
    """Calculator for Auction Participant track"""
    
    def __init__(self, track_params: TrackParameters):
        super().__init__(track_params)
        self.params: AuctionParticipantParameters = track_params.auction_participant
    
    def calculate_points(self) -> float:
        """Calculate points based on auction performance"""
        return self.params.calculate_points()
    
    def points_to_capital_equivalent(self, points: float) -> float:
        """Convert points to capital equivalent
        
        Auction participants' capital is their bid value adjusted by performance
        """
        # Points already factor in success rate and accuracy
        # Use average bid value as base
        if self.params.total_bids > 0:
            avg_bid_value = self.params.total_bid_value / self.params.total_bids
            performance_factor = (self.params.successful_bids / self.params.total_bids) * \
                               (self.params.bid_accuracy / 100)
            return avg_bid_value * (1 + performance_factor)
        return 0
    
    def get_risk_factor(self) -> float:
        """Auction participants have variable risk based on accuracy"""
        base_risk = 1.0
        if self.params.bid_accuracy < 50:
            return base_risk * 1.5  # Higher risk for poor accuracy
        elif self.params.bid_accuracy > 80:
            return base_risk * 0.8  # Lower risk for high accuracy
        return base_risk
    
    def get_allocation_model(self) -> str:
        """Auction participants benefit from tiered model based on performance"""
        return "tiered"


@dataclass
class MultiTrackResult:
    """Result of multi-track calculation"""
    track_results: Dict[TrackType, Dict[str, float]]
    total_points: float
    total_capital_equivalent: float
    weighted_risk_factor: float
    recommended_allocation: float
    allocation_breakdown: Dict[TrackType, float]


class MultiTrackCalculator:
    """Calculator for participants in multiple tracks"""
    
    def __init__(self, track_parameters_list: List[TrackParameters]):
        self.track_parameters = track_parameters_list
        self.calculators = self._create_calculators()
    
    def _create_calculators(self) -> Dict[TrackType, BaseTrackCalculator]:
        """Create appropriate calculator for each track"""
        calculators = {}
        for track_params in self.track_parameters:
            if track_params.track_type == TrackType.NODE_OPERATOR:
                calculators[track_params.track_type] = NodeOperatorCalculator(track_params)
            elif track_params.track_type == TrackType.RISK_UNDERWRITER:
                calculators[track_params.track_type] = RiskUnderwriterCalculator(track_params)
            elif track_params.track_type == TrackType.LIQUIDITY_PROVIDER:
                calculators[track_params.track_type] = LiquidityProviderCalculator(track_params)
            elif track_params.track_type == TrackType.AUCTION_PARTICIPANT:
                calculators[track_params.track_type] = AuctionParticipantCalculator(track_params)
        return calculators
    
    def calculate_multi_track_allocation(self, airdrop_params: AirdropParameters) -> MultiTrackResult:
        """Calculate allocation across multiple tracks"""
        track_results = {}
        total_points = 0
        total_capital_equivalent = 0
        risk_factors = []
        weights = []
        
        # Calculate for each track
        for track_type, calculator in self.calculators.items():
            points = calculator.calculate_points()
            capital_equiv = calculator.points_to_capital_equivalent(points)
            risk_factor = calculator.get_risk_factor()
            
            track_results[track_type] = {
                'points': points,
                'capital_equivalent': capital_equiv,
                'risk_factor': risk_factor,
                'allocation_model': calculator.get_allocation_model()
            }
            
            total_points += points
            total_capital_equivalent += capital_equiv
            risk_factors.append(risk_factor)
            weights.append(capital_equiv)
        
        # Calculate weighted risk factor
        if sum(weights) > 0:
            weighted_risk_factor = np.average(risk_factors, weights=weights)
        else:
            weighted_risk_factor = 1.0
        
        # Calculate total allocation using the dominant track's model
        # Find track with highest capital equivalent
        dominant_track = max(track_results.items(), 
                           key=lambda x: x[1]['capital_equivalent'])[0]
        dominant_model = track_results[dominant_track]['allocation_model']
        
        # Import the core calculator to use allocation estimation
        from .core import AirdropCalculator
        calc = AirdropCalculator(airdrop_params)
        
        # Use total capital equivalent for allocation calculation
        recommended_allocation = calc.estimate_user_allocation(
            total_capital_equivalent, 
            dominant_model
        )
        
        # Calculate allocation breakdown by track
        allocation_breakdown = {}
        if total_capital_equivalent > 0:
            for track_type, results in track_results.items():
                track_weight = results['capital_equivalent'] / total_capital_equivalent
                allocation_breakdown[track_type] = recommended_allocation * track_weight
        
        return MultiTrackResult(
            track_results=track_results,
            total_points=total_points,
            total_capital_equivalent=total_capital_equivalent,
            weighted_risk_factor=weighted_risk_factor,
            recommended_allocation=recommended_allocation,
            allocation_breakdown=allocation_breakdown
        )
    
    def get_track_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracks"""
        summary = {}
        for track_type, calculator in self.calculators.items():
            points = calculator.calculate_points()
            capital = calculator.points_to_capital_equivalent(points)
            risk = calculator.get_risk_factor()
            
            summary[track_type.name] = {
                'points': points,
                'capital_equivalent': capital,
                'risk_factor': risk,
                'model': calculator.get_allocation_model()
            }
        return summary


def create_track_calculator(track_params: TrackParameters) -> BaseTrackCalculator:
    """Factory function to create appropriate track calculator"""
    if track_params.track_type == TrackType.NODE_OPERATOR:
        return NodeOperatorCalculator(track_params)
    elif track_params.track_type == TrackType.RISK_UNDERWRITER:
        return RiskUnderwriterCalculator(track_params)
    elif track_params.track_type == TrackType.LIQUIDITY_PROVIDER:
        return LiquidityProviderCalculator(track_params)
    elif track_params.track_type == TrackType.AUCTION_PARTICIPANT:
        return AuctionParticipantCalculator(track_params)
    else:
        raise ValueError(f"Unknown track type: {track_params.track_type}")