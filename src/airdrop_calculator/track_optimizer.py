"""
Track optimization using Z3 solver for finding optimal track combinations
"""

from z3 import (
    Real, Bool, And, Or, If, Optimize, sat, Not, Sum
)
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from .types import TrackType, AirdropParameters, SolverConstraints
from .solver import EnhancedZ3Solver
from .tracks import create_track_calculator
from .defaults import TRACK_CAPITAL_REQUIREMENTS, OPTIMIZED_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class TrackOptimizationResult:
    """Result of track optimization"""
    track_weights: Dict[TrackType, float]
    expected_allocation: float
    total_capital_required: float
    weighted_risk_score: float
    diversity_bonus: float
    is_feasible: bool
    optimization_score: float


class TrackOptimizer:
    """Optimizer for finding optimal track participation strategies"""
    
    def __init__(self, solver: Optional[EnhancedZ3Solver] = None):
        self.solver = solver or EnhancedZ3Solver()
        
    def solve_optimal_track_combination(self, 
                                      user_profile: Dict,
                                      constraints: Optional[Dict] = None) -> TrackOptimizationResult:
        """
        Find optimal track weights for user profile using Z3
        
        Args:
            user_profile: Dict with keys:
                - available_capital: float
                - risk_tolerance: float (0.0 to 1.0)
                - time_horizon: int (months)
                - preferred_tracks: Optional[List[TrackType]]
            constraints: Optional additional constraints
        
        Returns:
            TrackOptimizationResult with optimal weights
        """
        logger.info("Starting track optimization with Z3...")
        
        opt = Optimize()
        
        # Define track weight variables (0 to 1)
        track_weights = {}
        for track_type in TrackType:
            weight = Real(f'{track_type.name}_weight')
            track_weights[track_type] = weight
            # Bound weights between 0 and 1
            opt.add(And(weight >= 0, weight <= 1))
        
        # Constraint: weights must sum to 1
        opt.add(Sum([w for w in track_weights.values()]) == 1.0)
        
        # Add user-specific constraints
        available_capital = user_profile.get('available_capital', 10000)
        risk_tolerance = user_profile.get('risk_tolerance', 0.5)
        time_horizon = user_profile.get('time_horizon', 12)
        
        # Capital constraints per track
        self._add_capital_constraints(opt, track_weights, available_capital)
        
        # Risk constraints
        self._add_risk_constraints(opt, track_weights, risk_tolerance)
        
        # Time horizon constraints
        self._add_time_constraints(opt, track_weights, time_horizon)
        
        # Add custom constraints if provided
        if constraints:
            self._add_custom_constraints(opt, track_weights, constraints)
        
        # Define objective function
        expected_allocation = self._calculate_expected_allocation(track_weights, user_profile)
        risk_score = self._calculate_risk_score(track_weights)
        diversity_bonus = self._calculate_diversity_bonus(track_weights)
        
        # Multi-objective: maximize allocation with risk penalty and diversity bonus
        objective = expected_allocation - (0.3 * risk_score) + (0.2 * diversity_bonus)
        opt.maximize(objective)
        
        # Solve
        if opt.check() == sat:
            model = opt.model()
            return self._extract_optimization_result(model, track_weights, user_profile)
        else:
            logger.warning("No feasible solution found for track optimization")
            return TrackOptimizationResult(
                track_weights={t: 0.0 for t in TrackType},
                expected_allocation=0.0,
                total_capital_required=0.0,
                weighted_risk_score=1.0,
                diversity_bonus=0.0,
                is_feasible=False,
                optimization_score=0.0
            )
    
    def _add_capital_constraints(self, opt: Optimize, track_weights: Dict, available_capital: float):
        """Add capital requirement constraints per track"""
        
        # Node operators need significant capital
        min_node_capital = TRACK_CAPITAL_REQUIREMENTS['NODE_OPERATOR']['base_capital_per_validator']
        opt.add(If(
            track_weights[TrackType.NODE_OPERATOR] > 0.01,
            available_capital >= min_node_capital,
            True
        ))
        
        # Limit node operator weight based on capital
        max_node_weight = min(1.0, available_capital / (5 * min_node_capital))
        opt.add(track_weights[TrackType.NODE_OPERATOR] <= max_node_weight)
        
        # Risk underwriters scale with available capital
        max_risk_weight = min(1.0, available_capital / 50000)
        opt.add(track_weights[TrackType.RISK_UNDERWRITER] <= max_risk_weight)
        
        # Auction participants need minimum capital
        min_auction_capital = TRACK_CAPITAL_REQUIREMENTS['AUCTION_PARTICIPANT']['min_bid_value']
        opt.add(If(
            track_weights[TrackType.AUCTION_PARTICIPANT] > 0.01,
            available_capital >= min_auction_capital * 10,  # 10x minimum for meaningful participation
            True
        ))
    
    def _add_risk_constraints(self, opt: Optimize, track_weights: Dict, risk_tolerance: float):
        """Add risk-based constraints"""
        
        # High-risk tracks limited by risk tolerance
        # Risk underwriter is highest risk (1.2x)
        opt.add(track_weights[TrackType.RISK_UNDERWRITER] <= risk_tolerance)
        
        # Auction participant is medium-high risk
        opt.add(track_weights[TrackType.AUCTION_PARTICIPANT] <= risk_tolerance * 1.2)
        
        # Conservative investors should have more in low-risk tracks
        if risk_tolerance < 0.3:
            # Favor liquidity provision (lowest risk)
            opt.add(track_weights[TrackType.LIQUIDITY_PROVIDER] >= 0.4)
    
    def _add_time_constraints(self, opt: Optimize, track_weights: Dict, time_horizon: int):
        """Add time horizon constraints"""
        
        # Short time horizons limit certain tracks
        if time_horizon < 6:
            # Risk underwriting requires longer commitment
            opt.add(track_weights[TrackType.RISK_UNDERWRITER] <= 0.2)
            # Node operation also requires time
            opt.add(track_weights[TrackType.NODE_OPERATOR] <= 0.3)
        
        # Long time horizons favor certain tracks
        if time_horizon >= 24:
            # Encourage risk underwriting for duration multiplier
            opt.add(track_weights[TrackType.RISK_UNDERWRITER] >= 0.1)
    
    def _add_custom_constraints(self, opt: Optimize, track_weights: Dict, constraints: Dict):
        """Add user-defined custom constraints"""
        
        # Minimum number of tracks
        if 'min_tracks' in constraints:
            min_tracks = constraints['min_tracks']
            # Count tracks with weight > 1%
            active_tracks = Sum([If(w > 0.01, 1, 0) for w in track_weights.values()])
            opt.add(active_tracks >= min_tracks)
        
        # Maximum concentration in any single track
        if 'max_concentration' in constraints:
            max_conc = constraints['max_concentration']
            for weight in track_weights.values():
                opt.add(weight <= max_conc)
        
        # Preferred tracks
        if 'preferred_tracks' in constraints:
            preferred = constraints['preferred_tracks']
            # Ensure at least 20% in each preferred track
            for track_type in preferred:
                if track_type in track_weights:
                    opt.add(track_weights[track_type] >= 0.2)
        
        # Excluded tracks
        if 'excluded_tracks' in constraints:
            excluded = constraints['excluded_tracks']
            for track_type in excluded:
                if track_type in track_weights:
                    opt.add(track_weights[track_type] == 0)
    
    def _calculate_expected_allocation(self, track_weights: Dict, user_profile: Dict) -> Real:
        """Calculate expected token allocation based on track weights"""
        
        available_capital = user_profile.get('available_capital', 10000)
        
        # Base allocations per track (simplified model)
        # These would be more complex in practice
        allocations = {
            TrackType.NODE_OPERATOR: 5000 * track_weights[TrackType.NODE_OPERATOR],
            TrackType.RISK_UNDERWRITER: 3000 * track_weights[TrackType.RISK_UNDERWRITER],
            TrackType.LIQUIDITY_PROVIDER: 4000 * track_weights[TrackType.LIQUIDITY_PROVIDER],
            TrackType.AUCTION_PARTICIPANT: 2000 * track_weights[TrackType.AUCTION_PARTICIPANT]
        }
        
        # Scale by available capital
        capital_multiplier = available_capital / 10000  # Normalized to 10k base
        
        total_allocation = Sum([
            alloc * capital_multiplier for alloc in allocations.values()
        ])
        
        return total_allocation
    
    def _calculate_risk_score(self, track_weights: Dict) -> Real:
        """Calculate weighted risk score"""
        
        # Risk factors per track
        risk_factors = {
            TrackType.NODE_OPERATOR: 0.8,
            TrackType.RISK_UNDERWRITER: 1.2,
            TrackType.LIQUIDITY_PROVIDER: 0.6,
            TrackType.AUCTION_PARTICIPANT: 1.0
        }
        
        weighted_risk = Sum([
            track_weights[track] * risk_factors[track] 
            for track in TrackType
        ])
        
        return weighted_risk
    
    def _calculate_diversity_bonus(self, track_weights: Dict) -> Real:
        """Calculate bonus for diversification across tracks"""
        
        # Count tracks with meaningful participation (>5%)
        active_tracks = Sum([
            If(weight > 0.05, 1, 0) 
            for weight in track_weights.values()
        ])
        
        # Diversity bonus increases with number of active tracks
        # 0 bonus for 1 track, up to 1.0 for 4 tracks
        diversity_bonus = active_tracks * 0.25
        
        return diversity_bonus
    
    def _extract_optimization_result(self, model, track_weights: Dict, 
                                   user_profile: Dict) -> TrackOptimizationResult:
        """Extract optimization result from Z3 model"""
        
        # Extract weights
        weights = {}
        total_weight = 0
        for track_type, weight_var in track_weights.items():
            weight_val = float(model.eval(weight_var).as_fraction())
            weights[track_type] = weight_val
            total_weight += weight_val
        
        # Normalize weights to ensure they sum to 1
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate metrics
        available_capital = user_profile.get('available_capital', 10000)
        
        # Expected allocation (simplified calculation)
        expected_allocation = sum([
            weights[track] * self._get_track_allocation_estimate(track, available_capital)
            for track in TrackType
        ])
        
        # Total capital required
        total_capital = sum([
            weights[track] * self._get_track_capital_requirement(track, available_capital)
            for track in TrackType
        ])
        
        # Weighted risk
        risk_factors = {
            TrackType.NODE_OPERATOR: 0.8,
            TrackType.RISK_UNDERWRITER: 1.2,
            TrackType.LIQUIDITY_PROVIDER: 0.6,
            TrackType.AUCTION_PARTICIPANT: 1.0
        }
        weighted_risk = sum([
            weights[track] * risk_factors[track]
            for track in TrackType
        ])
        
        # Diversity bonus
        active_tracks = sum(1 for w in weights.values() if w > 0.05)
        diversity_bonus = active_tracks * 0.25
        
        # Overall optimization score
        optimization_score = expected_allocation / (weighted_risk + 0.1) * (1 + diversity_bonus)
        
        return TrackOptimizationResult(
            track_weights=weights,
            expected_allocation=expected_allocation,
            total_capital_required=total_capital,
            weighted_risk_score=weighted_risk,
            diversity_bonus=diversity_bonus,
            is_feasible=True,
            optimization_score=optimization_score
        )
    
    def _get_track_allocation_estimate(self, track_type: TrackType, capital: float) -> float:
        """Estimate allocation for a track given capital"""
        
        # Simplified estimates - would be more complex in practice
        estimates = {
            TrackType.NODE_OPERATOR: capital * 3.0,  # 3x multiplier
            TrackType.RISK_UNDERWRITER: capital * 1.5,  # 1.5x multiplier
            TrackType.LIQUIDITY_PROVIDER: capital * 2.0,  # 2x multiplier
            TrackType.AUCTION_PARTICIPANT: capital * 1.2  # 1.2x multiplier
        }
        
        return estimates.get(track_type, capital)
    
    def _get_track_capital_requirement(self, track_type: TrackType, available: float) -> float:
        """Get capital requirement for a track"""
        
        # Use actual capital that would be deployed
        if track_type == TrackType.NODE_OPERATOR:
            return min(available * 0.8, 50000)  # High capital requirement
        elif track_type == TrackType.RISK_UNDERWRITER:
            return min(available * 0.5, 20000)  # Medium capital
        elif track_type == TrackType.LIQUIDITY_PROVIDER:
            return min(available * 0.6, 30000)  # Medium-high capital
        else:  # AUCTION_PARTICIPANT
            return min(available * 0.3, 10000)  # Lower capital
    
    def find_pareto_optimal_strategies(self, 
                                     user_profile: Dict,
                                     num_solutions: int = 10) -> List[TrackOptimizationResult]:
        """
        Find multiple Pareto-optimal track strategies
        
        Returns strategies that optimize different objectives:
        - Maximum allocation
        - Minimum risk
        - Maximum diversity
        - Balanced approach
        """
        logger.info(f"Finding {num_solutions} Pareto-optimal track strategies...")
        
        pareto_solutions = []
        
        # Define different objective weights to explore Pareto frontier
        objective_configs = [
            {'allocation': 1.0, 'risk': 0.0, 'diversity': 0.0},  # Max allocation
            {'allocation': 0.0, 'risk': 1.0, 'diversity': 0.0},  # Min risk
            {'allocation': 0.0, 'risk': 0.0, 'diversity': 1.0},  # Max diversity
            {'allocation': 0.5, 'risk': 0.3, 'diversity': 0.2},  # Balanced
            {'allocation': 0.7, 'risk': 0.2, 'diversity': 0.1},  # Allocation-focused
            {'allocation': 0.3, 'risk': 0.5, 'diversity': 0.2},  # Risk-averse
            {'allocation': 0.4, 'risk': 0.2, 'diversity': 0.4},  # Diversity-focused
            {'allocation': 0.6, 'risk': 0.3, 'diversity': 0.1},  # Moderate
            {'allocation': 0.2, 'risk': 0.6, 'diversity': 0.2},  # Conservative
            {'allocation': 0.8, 'risk': 0.1, 'diversity': 0.1},  # Aggressive
        ]
        
        for i, config in enumerate(objective_configs[:num_solutions]):
            # Modify user profile with objective weights
            modified_profile = user_profile.copy()
            modified_profile['objective_weights'] = config
            
            result = self.solve_optimal_track_combination(modified_profile)
            if result.is_feasible:
                pareto_solutions.append(result)
                logger.info(f"Found Pareto solution {i+1}: "
                          f"Allocation={result.expected_allocation:.0f}, "
                          f"Risk={result.weighted_risk_score:.2f}")
        
        return pareto_solutions
    
    def optimize_for_target_allocation(self,
                                     target_allocation: float,
                                     max_capital: float,
                                     risk_tolerance: float = 0.5) -> TrackOptimizationResult:
        """
        Find track combination that achieves target allocation with minimum capital
        """
        logger.info(f"Optimizing for target allocation: {target_allocation} tokens")
        
        # Binary search for minimum capital needed
        min_capital = 1000
        max_search_capital = max_capital
        best_result = None
        
        while max_search_capital - min_capital > 100:
            mid_capital = (min_capital + max_search_capital) / 2
            
            user_profile = {
                'available_capital': mid_capital,
                'risk_tolerance': risk_tolerance,
                'time_horizon': 12
            }
            
            result = self.solve_optimal_track_combination(user_profile)
            
            if result.is_feasible and result.expected_allocation >= target_allocation:
                best_result = result
                max_search_capital = mid_capital
            else:
                min_capital = mid_capital
        
        if best_result:
            logger.info(f"Found solution with capital: ${best_result.total_capital_required:.0f}")
            return best_result
        else:
            logger.warning("No feasible solution found for target allocation")
            return TrackOptimizationResult(
                track_weights={t: 0.0 for t in TrackType},
                expected_allocation=0.0,
                total_capital_required=max_capital,
                weighted_risk_score=1.0,
                diversity_bonus=0.0,
                is_feasible=False,
                optimization_score=0.0
            )