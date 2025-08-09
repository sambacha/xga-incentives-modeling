from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import numpy as np

class TrackType(Enum):
    """Types of tracks for multi-track airdrop system"""
    NODE_OPERATOR = auto()
    RISK_UNDERWRITER = auto()
    LIQUIDITY_PROVIDER = auto()
    AUCTION_PARTICIPANT = auto()

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class SolverError(Exception):
    """Custom exception for solver-related errors"""
    pass

@dataclass
class NodeOperatorParameters:
    """Parameters for Node Operator track"""
    validators_operated: int
    operation_duration_months: int
    validator_performance_score: float = 1.0
    uptime_percentage: float = 99.0
    
    def __post_init__(self):
        if not 1 <= self.validators_operated <= 100:
            raise ValidationError("Validators operated must be between 1 and 100")
        if not 1 <= self.operation_duration_months <= 36:
            raise ValidationError("Operation duration must be between 1 and 36 months")
        if not 0.8 <= self.validator_performance_score <= 1.2:
            raise ValidationError("Performance score must be between 0.8 and 1.2")
        if not 90 <= self.uptime_percentage <= 100:
            raise ValidationError("Uptime percentage must be between 90% and 100%")
    
    def calculate_points(self) -> float:
        """Calculate points for node operator track"""
        return (self.validators_operated * 
                self.operation_duration_months * 
                self.validator_performance_score * 
                (self.uptime_percentage / 100))

@dataclass
class RiskUnderwriterParameters:
    """Parameters for Risk Underwriter track"""
    tokens_staked: float
    staking_duration_months: int
    token_type: str = "FOLD"  # FOLD or EIGEN
    
    def __post_init__(self):
        if self.tokens_staked <= 0:
            raise ValidationError("Tokens staked must be positive")
        if not 1 <= self.staking_duration_months <= 48:
            raise ValidationError("Staking duration must be between 1 and 48 months")
        if self.token_type not in ["FOLD", "EIGEN"]:
            raise ValidationError("Token type must be FOLD or EIGEN")
    
    @property
    def duration_multiplier(self) -> float:
        """Calculate duration multiplier (1.0-2.0x based on staking duration)"""
        if self.staking_duration_months <= 6:
            return 1.0
        elif self.staking_duration_months <= 12:
            return 1.2
        elif self.staking_duration_months <= 24:
            return 1.5
        else:
            return 2.0
    
    def calculate_points(self) -> float:
        """Calculate points for risk underwriter track"""
        base_points = self.tokens_staked * (self.staking_duration_months / 12)
        return base_points * self.duration_multiplier

@dataclass
class LiquidityProviderParameters:
    """Parameters for Liquidity Provider track"""
    lst_amount: float
    liquidity_duration_months: int
    pool_type: str = "default"
    pool_bonus_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.lst_amount <= 0:
            raise ValidationError("LST amount must be positive")
        if not 1 <= self.liquidity_duration_months <= 36:
            raise ValidationError("Liquidity duration must be between 1 and 36 months")
        if not 1.0 <= self.pool_bonus_multiplier <= 1.5:
            raise ValidationError("Pool bonus multiplier must be between 1.0 and 1.5")
    
    def calculate_points(self) -> float:
        """Calculate points for liquidity provider track"""
        return (self.lst_amount * 
                (self.liquidity_duration_months / 12) * 
                self.pool_bonus_multiplier)

@dataclass
class AuctionParticipantParameters:
    """Parameters for Auction Participant track"""
    total_bids: int
    successful_bids: int
    bid_accuracy: float  # Average accuracy vs clearing price (0-100%)
    total_bid_value: float
    
    def __post_init__(self):
        if self.total_bids <= 0:
            raise ValidationError("Total bids must be positive")
        if self.successful_bids < 0:
            raise ValidationError("Successful bids cannot be negative")
        if self.successful_bids > self.total_bids:
            raise ValidationError("Successful bids cannot exceed total bids")
        if not 0 <= self.bid_accuracy <= 100:
            raise ValidationError("Bid accuracy must be between 0% and 100%")
        if self.total_bid_value <= 0:
            raise ValidationError("Total bid value must be positive")
    
    def calculate_points(self) -> float:
        """Calculate points for auction participant track"""
        success_rate = self.successful_bids / self.total_bids if self.total_bids > 0 else 0
        return self.total_bid_value * success_rate * (self.bid_accuracy / 100)

@dataclass
class TrackParameters:
    """Container for track-specific parameters"""
    track_type: TrackType
    node_operator: Optional[NodeOperatorParameters] = None
    risk_underwriter: Optional[RiskUnderwriterParameters] = None
    liquidity_provider: Optional[LiquidityProviderParameters] = None
    auction_participant: Optional[AuctionParticipantParameters] = None
    
    def __post_init__(self):
        """Ensure appropriate parameters are set for the track type"""
        if self.track_type == TrackType.NODE_OPERATOR and not self.node_operator:
            raise ValidationError("Node operator parameters required for NODE_OPERATOR track")
        elif self.track_type == TrackType.RISK_UNDERWRITER and not self.risk_underwriter:
            raise ValidationError("Risk underwriter parameters required for RISK_UNDERWRITER track")
        elif self.track_type == TrackType.LIQUIDITY_PROVIDER and not self.liquidity_provider:
            raise ValidationError("Liquidity provider parameters required for LIQUIDITY_PROVIDER track")
        elif self.track_type == TrackType.AUCTION_PARTICIPANT and not self.auction_participant:
            raise ValidationError("Auction participant parameters required for AUCTION_PARTICIPANT track")
    
    def calculate_points(self) -> float:
        """Calculate points based on track type"""
        if self.track_type == TrackType.NODE_OPERATOR:
            return self.node_operator.calculate_points()
        elif self.track_type == TrackType.RISK_UNDERWRITER:
            return self.risk_underwriter.calculate_points()
        elif self.track_type == TrackType.LIQUIDITY_PROVIDER:
            return self.liquidity_provider.calculate_points()
        elif self.track_type == TrackType.AUCTION_PARTICIPANT:
            return self.auction_participant.calculate_points()
        else:
            raise ValueError(f"Unknown track type: {self.track_type}")

@dataclass
class AirdropParameters:
    """Container for all airdrop parameters with validation"""
    total_supply: float
    airdrop_percent: float
    launch_price: float
    opportunity_cost: float
    volatility: float
    gas_cost: float
    campaign_duration: int
    airdrop_certainty: float
    revenue_share: float
    vesting_months: int
    immediate_unlock: float
    beta: Optional[float] = None
    hurdle_rate: Optional[float] = None
    penalties: Optional[Dict[str, float]] = None
    track_parameters: Optional[List[TrackParameters]] = None
    
    def __post_init__(self):
        """Validate all parameters on initialization"""
        # Basic range validations
        if self.total_supply <= 0:
            raise ValidationError("Total supply must be positive")
        if self.total_supply > 1e15:  # 1 quadrillion max
            raise ValidationError("Total supply exceeds maximum (1e15)")
        if not 0 < self.airdrop_percent <= 100:
            raise ValidationError("Airdrop percent must be between 0 and 100")
        if self.launch_price <= 0:
            raise ValidationError("Launch price must be positive")
        if self.launch_price > 1e6:  # $1M max per token
            raise ValidationError("Launch price exceeds maximum ($1M)")
        if not 2 <= self.opportunity_cost <= 100:
            raise ValidationError("Opportunity cost must be between 2% and 100%")
        if not 10 <= self.volatility <= 200:
            raise ValidationError("Volatility must be between 10% and 200%")
        if self.gas_cost < 0:
            raise ValidationError("Gas cost cannot be negative")
        if self.gas_cost > 10000:  # $10k max gas cost
            raise ValidationError("Gas cost exceeds maximum ($10k)")
        if not 1 <= self.campaign_duration <= 36:
            raise ValidationError("Campaign duration must be between 1 and 36 months")
        if not 0 <= self.airdrop_certainty <= 100:
            raise ValidationError("Airdrop certainty must be between 0% and 100%")
        if not 0 <= self.revenue_share <= 100:
            raise ValidationError("Revenue share must be between 0% and 100%")
        if self.vesting_months < 0:
            raise ValidationError("Vesting months cannot be negative")
        if self.vesting_months > 120:  # 10 years max
            raise ValidationError("Vesting months exceeds maximum (120 months)")
        if not 0 <= self.immediate_unlock <= 100:
            raise ValidationError("Immediate unlock must be between 0% and 100%")
        
        # Invariant checks for option pricing parameters
        if self.beta is not None:
            if self.beta <= 1.0:
                raise ValidationError(f"Beta must be > 1.0 for valid option pricing, got {self.beta}")
            if self.beta > 10.0:
                raise ValidationError(f"Beta exceeds reasonable maximum (10.0), got {self.beta}")
        
        if self.hurdle_rate is not None:
            if self.hurdle_rate < 1.0:
                raise ValidationError(f"Hurdle rate must be >= 1.0 (100% return minimum), got {self.hurdle_rate}")
            if self.hurdle_rate > 20.0:
                raise ValidationError(f"Hurdle rate exceeds reasonable maximum (20.0), got {self.hurdle_rate}")
            
            # Cross-validation: if both beta and hurdle_rate exist, verify relationship
            if self.beta is not None:
                expected_hurdle = self.beta / (self.beta - 1)
                # Allow 1% tolerance for numerical errors
                if abs(expected_hurdle - self.hurdle_rate) / self.hurdle_rate > 0.01:
                    raise ValidationError(
                        f"Inconsistent beta ({self.beta}) and hurdle_rate ({self.hurdle_rate}). "
                        f"Expected hurdle_rate = {expected_hurdle:.4f}"
                    )
        
        # Check for numerical overflow in market cap calculation
        market_cap = self.total_supply * self.launch_price
        if market_cap > 1e18:  # 1 quintillion max market cap
            raise ValidationError(f"Market cap ({market_cap}) exceeds maximum (1e18)")
        
        # Validate probabilities are proper
        if self.airdrop_certainty == 0:
            import logging
            logging.getLogger(__name__).warning("Airdrop certainty is 0%, making all calculations undefined")

@dataclass
class UserSegment:
    """Represents a user segment with validation"""
    name: str
    min_capital: float
    max_capital: float
    population_percent: float
    avg_transactions: int = 10
    
    def __post_init__(self):
        if self.min_capital < 0:
            raise ValidationError("Minimum capital cannot be negative")
        if self.max_capital <= self.min_capital:
            raise ValidationError("Maximum capital must be greater than minimum capital")
        if not 0 <= self.population_percent <= 100:
            raise ValidationError("Population percent must be between 0 and 100")
        if self.avg_transactions < 0:
            raise ValidationError("Average transactions cannot be negative")

@dataclass
class SegmentResult:
    """Result of segment profitability analysis"""
    segment: str
    avg_capital: float
    min_tokens: float
    estimated_allocation: float
    profitable: bool
    roi: float
    population_percent: float

@dataclass
class MarketMetrics:
    """Comprehensive market metrics result"""
    min_market_cap: float
    hurdle_rate: float
    typical_user_break_even: float
    profitable_users_percent: float
    avg_roi: float
    optimal_capital: float
    segment_results: List[SegmentResult]
    beta_value: float
    required_return_multiple: float

@dataclass
class Scenario:
    """Defines a scenario to be evaluated"""
    name: str
    description: str
    constraints: Dict[str, Union[float, int]]
    objectives: Dict[str, Dict[str, Union[float, str]]]

@dataclass
class SolverConstraints:
    """Constraints for Z3 solver"""
    min_supply: Optional[float] = None
    max_supply: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_airdrop_percent: Optional[float] = 5
    max_airdrop_percent: Optional[float] = 50
    opportunity_cost: Optional[float] = None
    volatility: Optional[float] = None
    gas_cost: Optional[float] = None
    campaign_duration: Optional[int] = None
    
    def __post_init__(self):
        """Validate constraints"""
        if self.min_supply and self.max_supply and self.min_supply > self.max_supply:
            raise ValidationError("Minimum supply cannot exceed maximum supply")
        if self.min_price and self.max_price and self.min_price > self.max_price:
            raise ValidationError("Minimum price cannot exceed maximum price")
