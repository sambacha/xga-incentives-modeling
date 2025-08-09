from z3 import (
    And, If, Int, Optimize, Real, Solver, sat, Tactic
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import hashlib
import json
import concurrent.futures
import time
from dataclasses import dataclass

from .types import AirdropParameters, SolverConstraints, SolverError

logger = logging.getLogger(__name__)


@dataclass
class ConstraintState:
    """State representation for Kalman filter tracking constraint effectiveness"""
    relaxation_level: float  # How much constraints are relaxed (0=strict, 1=fully relaxed)
    success_probability: float  # Estimated probability of finding solution
    solve_time: float  # Expected solving time
    solution_quality: float  # Quality of solution when found


class KalmanConstraintEstimator:
    """
    Kalman filter for adaptively estimating optimal constraint relaxation strategies.
    
    State vector: [relaxation_level, success_prob, solve_time, quality]
    Observations: [actual_success, actual_time, actual_quality]
    """
    
    def __init__(self, dim_state=4, dim_obs=3):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        
        # State: [relaxation_level, success_prob, solve_time, quality]
        self.x = np.array([0.5, 0.5, 10.0, 0.5])  # Initial state
        
        # State covariance (uncertainty in our estimates)
        self.P = np.eye(dim_state) * 0.1
        
        # Process noise (how much we expect state to change between steps)
        self.Q = np.eye(dim_state) * 0.01
        
        # Observation noise (measurement uncertainty)
        self.R = np.eye(dim_obs) * 0.05
        
        # State transition model (how state evolves)
        self.F = np.eye(dim_state)
        self.F[0, 1] = 0.1  # Success prob influences next relaxation
        self.F[2, 0] = 2.0  # Relaxation affects solve time
        
        # Observation model (how observations relate to state)
        self.H = np.zeros((dim_obs, dim_state))
        self.H[0, 1] = 1.0  # Observe success probability
        self.H[1, 2] = 1.0  # Observe solve time
        self.H[2, 3] = 1.0  # Observe solution quality
        
        # History for adaptive learning
        self.history = []
        
    def predict(self) -> np.ndarray:
        """Predict next state using process model"""
        # Check for corrupted state and reset if needed
        if not np.all(np.isfinite(self.x)):
            logger.warning("Corrupted Kalman state detected, resetting to defaults")
            self.x = np.array([0.5, 0.5, 10.0, 0.5])
            self.P = np.eye(self.dim_state) * 0.1
        
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Limit covariance to prevent explosion
        max_variance = 100.0  # Maximum allowed variance
        for i in range(self.dim_state):
            if self.P[i, i] > max_variance:
                self.P[i, i] = max_variance
        
        # Ensure state remains valid after prediction
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)  # Relaxation level
        self.x[1] = np.clip(self.x[1], 0.0, 1.0)  # Success probability
        self.x[2] = np.clip(self.x[2], 0.1, 60.0)  # Solve time
        self.x[3] = np.clip(self.x[3], 0.0, 1.0)  # Quality
        
        return self.x.copy()
    
    def update(self, observation: np.ndarray, success: bool) -> np.ndarray:
        """Update state estimate with new observation"""
        # If we have a success, include it in observation
        if success:
            z = observation
        else:
            # Failed solve: set success=0, high time penalty, low quality
            z = np.array([0.0, min(observation[1], 30.0), 0.1])
        
        # Kalman gain calculation with improved numerical stability
        S = self.H @ self.P @ self.H.T + self.R
        
        # Check condition number for numerical stability
        cond_num = np.linalg.cond(S)
        if cond_num > 1e10:
            logger.warning(f"High condition number in Kalman filter: {cond_num}")
            # Use regularized inverse
            S_reg = S + np.eye(self.dim_obs) * 1e-6
            try:
                S_inv = np.linalg.inv(S_reg)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if regularization fails
                S_inv = np.linalg.pinv(S)
        else:
            # Use standard inverse for well-conditioned matrices
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                S_inv = np.linalg.pinv(S)
        
        K = self.P @ self.H.T @ S_inv
        
        # State update
        y = z - self.H @ self.x  # Innovation
        self.x = self.x + K @ y
        
        # Covariance update using Joseph form for numerical stability
        # P = (I - KH)P(I - KH)' + KRK'
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Clip state to reasonable bounds
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)  # Relaxation level
        self.x[1] = np.clip(self.x[1], 0.0, 1.0)  # Success probability
        self.x[2] = np.clip(self.x[2], 0.1, 60.0)  # Solve time
        self.x[3] = np.clip(self.x[3], 0.0, 1.0)  # Quality
        
        # Store history
        self.history.append({
            'observation': z.copy(),
            'state': self.x.copy(),
            'success': success
        })
        
        return self.x.copy()
    
    def get_optimal_relaxation(self) -> float:
        """Get the current optimal relaxation level"""
        return self.x[0]
    
    def get_success_probability(self) -> float:
        """Get estimated success probability at current relaxation"""
        return self.x[1]
    
    def get_expected_time(self) -> float:
        """Get expected solve time at current relaxation"""
        return self.x[2]
    
    def adapt_process_noise(self):
        """Adaptively adjust process noise based on recent performance"""
        if len(self.history) < 5:
            return
            
        # Calculate variance in recent observations
        recent_obs = [h['observation'] for h in self.history[-5:]]
        recent_states = [h['state'] for h in self.history[-5:]]
        
        obs_var = np.var(recent_obs, axis=0)
        state_var = np.var(recent_states, axis=0)
        
        # Adjust Q based on observed variability
        self.Q = np.diag(np.clip(state_var * 0.1, 0.001, 0.1))


class AdaptiveConstraintScheduler:
    """
    Schedules constraint relaxation levels using Kalman filter predictions
    """
    
    def __init__(self, initial_constraints: SolverConstraints):
        self.base_constraints = initial_constraints
        self.kalman = KalmanConstraintEstimator()
        self.relaxation_history = []
        
    def get_next_constraint_level(self) -> Tuple[float, SolverConstraints]:
        """Get next constraint relaxation level based on Kalman predictions"""
        # Predict optimal relaxation level
        predicted_state = self.kalman.predict()
        relaxation_level = predicted_state[0]
        
        # Generate relaxed constraints
        relaxed_constraints = self._relax_constraints(relaxation_level)
        
        return relaxation_level, relaxed_constraints
    
    def _relax_constraints(self, level: float) -> SolverConstraints:
        """Create relaxed constraints based on relaxation level (0=strict, 1=fully relaxed)"""
        # Start with base constraints
        base = self.base_constraints
        
        # Calculate relaxed bounds
        supply_range = (base.max_supply or 10_000_000_000) - (base.min_supply or 1_000_000)
        price_range = (base.max_price or 10.0) - (base.min_price or 0.01)
        airdrop_range = base.max_airdrop_percent - base.min_airdrop_percent
        
        # Apply relaxation
        relaxed = SolverConstraints(
            min_supply=max(1_000_000, (base.min_supply or 1_000_000) - level * supply_range * 0.2),
            max_supply=min(50_000_000_000, (base.max_supply or 10_000_000_000) + level * supply_range * 0.2),
            min_price=max(0.001, (base.min_price or 0.01) - level * price_range * 0.3),
            max_price=min(100.0, (base.max_price or 10.0) + level * price_range * 0.3),
            min_airdrop_percent=max(1.0, base.min_airdrop_percent - level * airdrop_range * 0.4),
            max_airdrop_percent=min(80.0, base.max_airdrop_percent + level * airdrop_range * 0.4)
        )
        
        return relaxed
    
    def update_performance(self, success: bool, solve_time: float, solution_quality: float):
        """Update Kalman filter with observed performance"""
        observation = np.array([float(success), solve_time, solution_quality])
        self.kalman.update(observation, success)
        
        # Adaptive learning
        if len(self.kalman.history) % 10 == 0:
            self.kalman.adapt_process_noise()
        
        self.relaxation_history.append({
            'relaxation_level': self.kalman.get_optimal_relaxation(),
            'success': success,
            'solve_time': solve_time,
            'quality': solution_quality
        })
    
    def get_performance_summary(self) -> Dict:
        """Get summary of adaptive performance"""
        if not self.relaxation_history:
            return {}
            
        recent = self.relaxation_history[-10:]
        return {
            'total_attempts': len(self.relaxation_history),
            'recent_success_rate': sum(h['success'] for h in recent) / len(recent),
            'avg_solve_time': np.mean([h['solve_time'] for h in recent]),
            'avg_quality': np.mean([h['quality'] for h in recent]),
            'current_relaxation': self.kalman.get_optimal_relaxation(),
            'predicted_success_prob': self.kalman.get_success_probability()
        }


def _solve_pareto_point_worker(objectives: List[str], constraints_dict: Dict, epsilon: float) -> Optional[Dict]:
    """
    Worker function to solve for a single point on the Pareto frontier.
    This function is designed to be called in a separate process.
    """
    opt = Optimize()

    # --- Replicated from _define_variables ---
    variables = {
        'total_supply': Real('total_supply'),
        'airdrop_percent': Real('airdrop_percent'),
        'launch_price': Real('launch_price'),
        'opportunity_cost': Real('opportunity_cost'),
        'volatility': Real('volatility'),
        'gas_cost': Real('gas_cost'),
        'campaign_duration': Int('campaign_duration'),
        'airdrop_certainty': Real('airdrop_certainty')
    }

    # --- Replicated from _add_basic_constraints ---
    constraints = SolverConstraints(**constraints_dict)
    min_supply = constraints.min_supply if constraints.min_supply is not None else 1_000_000
    max_supply = constraints.max_supply if constraints.max_supply is not None else 10_000_000_000
    opt.add(variables['total_supply'] >= min_supply, variables['total_supply'] <= max_supply)
    
    opt.add(variables['airdrop_percent'] >= constraints.min_airdrop_percent, variables['airdrop_percent'] <= constraints.max_airdrop_percent)
    
    min_price = constraints.min_price if constraints.min_price is not None else 0.01
    max_price = constraints.max_price if constraints.max_price is not None else 10.0
    opt.add(variables['launch_price'] >= min_price, variables['launch_price'] <= max_price)
    
    opt.add(variables['opportunity_cost'] >= 2, variables['opportunity_cost'] <= 50)
    opt.add(variables['volatility'] >= 30, variables['volatility'] <= 150)
    opt.add(variables['gas_cost'] >= 10, variables['gas_cost'] <= 500)
    opt.add(variables['campaign_duration'] >= 3, variables['campaign_duration'] <= 24)
    opt.add(variables['airdrop_certainty'] >= 50, variables['airdrop_certainty'] <= 100)
    
    if constraints.opportunity_cost is not None:
        opt.add(variables['opportunity_cost'] == constraints.opportunity_cost)
    if constraints.volatility is not None:
        opt.add(variables['volatility'] == constraints.volatility)
    if constraints.gas_cost is not None:
        opt.add(variables['gas_cost'] == constraints.gas_cost)
    if constraints.campaign_duration is not None:
        opt.add(variables['campaign_duration'] == constraints.campaign_duration)

    # --- Replicated from _create_option_pricing_expressions ---
    r = variables['opportunity_cost'] / 100
    sigma = variables['volatility'] / 100
    sigma_squared = sigma * sigma

    a = 0.5 - (r - 0.0) / sigma_squared
    discriminant_val = a * a + 2 * r / sigma_squared
    
    # Create unique variable name for this worker process
    sqrt_var_name = f"sqrt_disc_worker_{abs(hash((str(epsilon), str(objectives))))}"
    sqrt_discriminant = Real(sqrt_var_name)
    
    beta = a + sqrt_discriminant
    
    # Use robust hurdle rate calculation
    epsilon_z3 = 0.001
    hurdle_rate = If(
        beta > 1.0 + epsilon_z3,
        beta / (beta - 1),
        1.5  # Safe default
    )
    
    constraints_list = [
        sqrt_discriminant * sqrt_discriminant == discriminant_val,
        sqrt_discriminant >= 0,
        beta >= 1.01,
        hurdle_rate >= 1.1,
        hurdle_rate <= 10.0
    ]
    option_expressions = {
        "beta": beta,
        "hurdle_rate": hurdle_rate,
        "constraints": constraints_list
    }
    opt.add(option_expressions['constraints'])

    # --- Logic from _solve_pareto_point ---
    objective_funcs = {
        'market_cap': variables['total_supply'] * variables['launch_price'],
        'profitability': (30 - variables['opportunity_cost']) * 2 + variables['airdrop_percent'],
        'hurdle_rate': option_expressions['hurdle_rate'],
    }
    
    optimization_direction = {
        'market_cap': 'maximize',
        'profitability': 'maximize',
        'hurdle_rate': 'minimize',
    }

    primary_obj, secondary_obj = objectives[0], objectives[1]

    if optimization_direction[secondary_obj] == 'maximize':
        opt.add(objective_funcs[secondary_obj] >= epsilon)
    else:
        opt.add(objective_funcs[secondary_obj] <= epsilon)

    if optimization_direction[primary_obj] == 'maximize':
        opt.maximize(objective_funcs[primary_obj])
    else:
        opt.minimize(objective_funcs[primary_obj])

    if opt.check() == sat:
        model = opt.model()
        # --- Replicated from _extract_solution_dict ---
        solution = {
            'total_supply': float(model[variables['total_supply']].as_decimal(9)),
            'airdrop_percent': float(model[variables['airdrop_percent']].as_decimal(9)),
            'launch_price': float(model[variables['launch_price']].as_decimal(9)),
            'opportunity_cost': float(model[variables['opportunity_cost']].as_decimal(9)),
            'volatility': float(model[variables['volatility']].as_decimal(9)),
            'gas_cost': float(model[variables['gas_cost']].as_decimal(9)),
            'campaign_duration': int(model[variables['campaign_duration']].as_long()),
            'airdrop_certainty': float(model[variables['airdrop_certainty']].as_decimal(9))
        }
        solution[f'{primary_obj}_value'] = float(model.eval(objective_funcs[primary_obj]).as_decimal(9))
        solution[f'{secondary_obj}_value'] = float(model.eval(objective_funcs[secondary_obj]).as_decimal(9))
        return solution
    return None

class EnhancedZ3Solver:
    """Enhanced Z3 solver with advanced features"""
    
    def __init__(self):
        self.optimizer = Optimize()
        self._solution_cache = {}

    def _get_cache_key(self, method_name: str, *args, **kwargs) -> str:
        """Creates a stable hash for the given arguments."""
        # This is a simplified example. A robust implementation
        # would need to handle the structure of your input objects.
        s = json.dumps((method_name, args, kwargs), sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()

    def solve_with_nonlinear_constraints(self,
                                         target_market_cap: float,
                                         target_profitable_users: float,
                                         constraints: SolverConstraints) -> Optional[AirdropParameters]:
        """
        Solves for airdrop parameters by modeling the problem as a random-start
        perpetual American option, using Z3 tactics for non-linear optimization.

        This method leverages the mathematical framework from "Retroactive Airdrops
        are Exotic Options" to create a robust valuation model.
        """
        cache_key = self._get_cache_key('solve_with_nonlinear_constraints', target_market_cap, target_profitable_users, constraints)
        if cache_key in self._solution_cache:
            logger.info("Returning cached solution.")
            return self._solution_cache[cache_key]

        tactic = Tactic('qfnra-nlsat')
        solver = tactic.solver()

        variables = self._define_variables()
        self._add_basic_constraints(solver, **variables, constraints=constraints)

        option_expressions = self._create_option_pricing_expressions(
            variables['opportunity_cost'], variables['volatility']
        )
        solver.add(option_expressions['constraints'])
        
        beta = option_expressions['beta']
        hurdle_rate = option_expressions['hurdle_rate']
        # Beta constraint is already enforced in option_expressions (beta >= 1.2)
        # Hurdle rate constraint - allow wider range for solutions
        solver.add(If(beta > 1.01, hurdle_rate <= 10.0, True))

        market_cap = variables['total_supply'] * variables['launch_price']
        solver.add(market_cap >= target_market_cap * 0.9)
        solver.add(market_cap <= target_market_cap * 1.1)

        profitability_proxy = (30 - variables['opportunity_cost']) * 2 + variables['airdrop_percent']
        solver.add(profitability_proxy >= target_profitable_users * 0.8)

        complexity_score = self._estimate_complexity(solver, len(variables))
        estimated_time = self._estimate_time(complexity_score)
        logger.info(f"Estimated solving time: ~{estimated_time:.2f} seconds")

        solver.set("timeout", 30000)

        if solver.check() == sat:
            model = solver.model()
            solution = self._extract_solution(model, variables, option_expressions)
            self._solution_cache[cache_key] = solution
            return solution

        self._solution_cache[cache_key] = None
        return None
    
    def _estimate_complexity(self, solver, num_vars: int) -> float:
        """Estimate complexity score based on solver properties"""
        num_constraints = len(solver.assertions())
        w_vars = 1.5
        w_constraints = 1.0
        score = (num_vars * w_vars) + (num_constraints * w_constraints)
        return score

    def _estimate_time(self, score: float) -> float:
        """Convert complexity score to an estimated time in seconds"""
        return score * 0.05

    def solve_with_soft_constraints(self, 
                                  objectives: Dict[str, Tuple[float, float]],
                                  constraints: SolverConstraints) -> Optional[AirdropParameters]:
        """
        Solve with soft constraints and weighted objectives
        objectives: Dict of {param_name: (target_value, weight)}
        """
        cache_key = self._get_cache_key('solve_with_soft_constraints', objectives, constraints)
        if cache_key in self._solution_cache:
            logger.info("Returning cached solution.")
            return self._solution_cache[cache_key]

        opt = Optimize()
        variables = self._define_variables()
        self._add_basic_constraints(opt, **variables, constraints=constraints)
        
        # Don't add option pricing constraints to avoid over-constraining the problem
        # We'll compute beta and hurdle_rate numerically after solving
        
        penalties = {}
        penalty_expressions = []
        for param_name, (target, weight) in objectives.items():
            if param_name == 'market_cap':
                market_cap = variables['total_supply'] * variables['launch_price']
                diff = If(market_cap > target, market_cap - target, target - market_cap)
                penalty_expr = weight * diff / (target or 1.0)
            elif param_name == 'profitable_users':
                profitability = (30 - variables['opportunity_cost']) * 2 + variables['airdrop_percent']
                diff = If(profitability > target, profitability - target, target - profitability)
                penalty_expr = weight * diff / 100
            elif param_name in variables:
                var = variables[param_name]
                diff = If(var > target, var - target, target - var)
                penalty_expr = weight * diff / (target or 1.0)
            
            penalties[param_name] = penalty_expr
            penalty_expressions.append(penalty_expr)

        opt.minimize(sum(penalty_expressions))
        
        complexity_score = self._estimate_complexity(opt, len(variables))
        estimated_time = self._estimate_time(complexity_score)
        logger.info(f"Estimated solving time: ~{estimated_time:.2f} seconds")
        
        # Set timeout to prevent infinite hangs
        opt.set("timeout", 5000)  # 5 seconds
        
        if opt.check() == sat:
            model = opt.model()
            evaluated_penalties = {
                name: float(model.eval(expr).as_decimal(5).replace('?', ''))
                for name, expr in penalties.items()
            }
            solution = self._extract_solution(model, variables, None, evaluated_penalties)
            self._solution_cache[cache_key] = solution
            return solution
        
        self._solution_cache[cache_key] = None
        return None
    
    def solve_incremental_with_relaxation(self,
                                        target_market_cap: float,
                                        target_profitable_users: float,
                                        constraint_levels: List[Tuple[int, SolverConstraints]]) -> Optional[AirdropParameters]:
        """
        Incrementally relaxes constraints using Z3's push/pop for efficiency.
        """
        sorted_levels = sorted(constraint_levels, key=lambda x: x[0], reverse=True)
        
        solver = Solver()
        variables = self._define_variables()
        
        # Add base constraints that are always active
        self._add_basic_constraints(solver, **variables, constraints=sorted_levels[0][1])

        for i in range(len(sorted_levels)):
            solver.push()
            
            _, constraints = sorted_levels[i]
            self._add_basic_constraints(solver, **variables, constraints=constraints)
            
            # Don't add option pricing constraints to avoid over-constraining
            
            market_cap = variables['total_supply'] * variables['launch_price']
            solver.add(And(market_cap >= target_market_cap * 0.9, market_cap <= target_market_cap * 1.1))
            
            complexity_score = self._estimate_complexity(solver, len(variables))
            estimated_time = self._estimate_time(complexity_score)
            logger.info(f"Estimated solving time for level {sorted_levels[i][0]}: ~{estimated_time:.2f} seconds")
            
            # Set timeout to prevent infinite hangs
            solver.set("timeout", 5000)  # 5 seconds
            
            if solver.check() == sat:
                logger.info(f"Solution found at constraint level {sorted_levels[i][0]}")
                model = solver.model()
                return self._extract_solution(model, variables, None)
            
            solver.pop()
        
        logger.warning("No solution found even after relaxing all constraints")
        return None
    
    def solve_incremental_with_kalman(self,
                                    target_market_cap: float,
                                    target_profitable_users: float,
                                    initial_constraints: SolverConstraints,
                                    max_iterations: int = 20) -> Optional[AirdropParameters]:
        """
        Kalman filter-enhanced incremental solver that adaptively learns optimal 
        constraint relaxation strategies.
        
        The Kalman filter tracks:
        - Optimal relaxation levels
        - Success probability predictions  
        - Expected solve times
        - Solution quality estimates
        
        This leads to faster convergence and better solutions by learning from
        previous solve attempts.
        """
        # Validate inputs
        if target_market_cap <= 0:
            raise ValueError(f"Target market cap must be positive, got {target_market_cap}")
        if target_profitable_users <= 0 or target_profitable_users > 100:
            raise ValueError(f"Target profitable users must be between 0 and 100%, got {target_profitable_users}")
        if max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive, got {max_iterations}")
        
        logger.info("Starting Kalman-enhanced incremental solving...")
        
        # Initialize adaptive scheduler
        scheduler = AdaptiveConstraintScheduler(initial_constraints)
        
        best_solution = None
        best_quality = 0.0
        
        for iteration in range(max_iterations):
            logger.info(f"Kalman iteration {iteration + 1}/{max_iterations}")
            
            # Get next constraint level from Kalman predictor
            relaxation_level, constraints = scheduler.get_next_constraint_level()
            
            logger.info(f"Trying relaxation level: {relaxation_level:.3f}")
            logger.info(f"Predicted success probability: {scheduler.kalman.get_success_probability():.3f}")
            logger.info(f"Expected solve time: {scheduler.kalman.get_expected_time():.1f}s")
            
            # Attempt to solve with current constraints
            start_time = time.time()
            expected_time = scheduler.kalman.get_expected_time()
            solution = self._solve_with_constraints(target_market_cap, target_profitable_users, constraints, expected_time)
            solve_time = time.time() - start_time
            
            # Calculate solution quality
            if solution:
                quality = self._calculate_solution_quality(solution, target_market_cap, target_profitable_users)
                success = True
                
                # Keep track of best solution
                if quality > best_quality:
                    best_solution = solution
                    best_quality = quality
                    
                logger.info(f"✓ Solution found! Quality: {quality:.3f}, Time: {solve_time:.2f}s")
                
                # Early termination if we find a high-quality solution
                if quality > 0.9:
                    logger.info("High-quality solution found, terminating early")
                    break
                    
            else:
                quality = 0.0
                success = False
                logger.info(f"✗ No solution found, Time: {solve_time:.2f}s")
            
            # Update Kalman filter with observed performance
            scheduler.update_performance(success, solve_time, quality)
            
            # Log performance summary every 5 iterations
            if (iteration + 1) % 5 == 0:
                summary = scheduler.get_performance_summary()
                logger.info(f"Performance summary: {summary}")
        
        # Final performance report
        final_summary = scheduler.get_performance_summary()
        logger.info(f"Final Kalman performance: {final_summary}")
        
        if best_solution:
            logger.info(f"Best solution found with quality {best_quality:.3f}")
            return best_solution
        else:
            logger.warning("No solution found with Kalman-enhanced approach")
            return None
    
    def _solve_with_constraints(self, 
                               target_market_cap: float,
                               target_profitable_users: float, 
                               constraints: SolverConstraints,
                               expected_time: float = 10.0) -> Optional[AirdropParameters]:
        """Solve with given constraints (helper for Kalman method)"""
        solver = Solver()
        variables = self._define_variables()
        
        # Add constraints
        self._add_basic_constraints(solver, **variables, constraints=constraints)
        
        # Add target constraints
        market_cap = variables['total_supply'] * variables['launch_price']
        solver.add(And(
            market_cap >= target_market_cap * 0.9, 
            market_cap <= target_market_cap * 1.1
        ))
        
        # Set timeout based on expected time
        solver.set("timeout", max(5000, int(expected_time * 1000)))
        
        if solver.check() == sat:
            model = solver.model()
            return self._extract_solution(model, variables, None)
        else:
            return None
    
    def _calculate_solution_quality(self, 
                                  solution: AirdropParameters,
                                  target_market_cap: float, 
                                  target_profitable_users: float) -> float:
        """Calculate solution quality score (0-1)"""
        quality_factors = []
        
        # Market cap proximity (closer to target = higher quality)
        actual_market_cap = solution.total_supply * solution.launch_price
        market_cap_error = abs(actual_market_cap - target_market_cap) / target_market_cap
        market_cap_quality = max(0, 1 - market_cap_error)
        quality_factors.append(market_cap_quality * 0.4)
        
        # Parameter reasonableness (realistic values = higher quality)
        param_quality = 1.0
        
        # Penalize extreme opportunity cost
        if solution.opportunity_cost > 100:
            param_quality *= 0.1  # Severe penalty for very extreme values
        elif solution.opportunity_cost < 2 or solution.opportunity_cost > 50:
            param_quality *= 0.8
            
        if solution.volatility < 30 or solution.volatility > 150:
            param_quality *= 0.8
        if solution.airdrop_percent < 5 or solution.airdrop_percent > 50:
            param_quality *= 0.8
        quality_factors.append(param_quality * 0.3)
        
        # Economic viability (profitable scenarios = higher quality)
        if solution.hurdle_rate and 1 < solution.hurdle_rate < 5:
            economic_quality = 1.0
        elif solution.hurdle_rate:
            economic_quality = max(0, 1 - abs(solution.hurdle_rate - 2.5) / 10)
        else:
            economic_quality = 0.5
        quality_factors.append(economic_quality * 0.3)
        
        return sum(quality_factors)
    
    def find_pareto_optimal_solutions(self,
                                    objectives: List[str],
                                    constraints: SolverConstraints,
                                    num_solutions: int = 10) -> List[Dict]:
        """
        Finds Pareto-optimal solutions by parallelizing the Epsilon-Constraint method
        across multiple CPU cores.
        """
        if len(objectives) != 2:
            raise ValueError("Epsilon-Constraint method currently supports exactly two objectives.")

        variables = self._define_variables()
        option_expressions = self._create_option_pricing_expressions(
            variables['opportunity_cost'], variables['volatility']
        )
        
        objective_funcs = {
            'market_cap': variables['total_supply'] * variables['launch_price'],
            'profitability': (30 - variables['opportunity_cost']) * 2 + variables['airdrop_percent'],
            'hurdle_rate': option_expressions['hurdle_rate'],
        }
        
        primary_obj, secondary_obj = objectives[0], objectives[1]
        min_secondary_val, max_secondary_val = self._get_objective_bounds(secondary_obj, objective_funcs, variables, constraints)

        if min_secondary_val is None or max_secondary_val is None:
            logger.error("Could not determine bounds for the secondary objective.")
            return []

        pareto_solutions = []
        epsilon_values = np.linspace(min_secondary_val, max_secondary_val, num_solutions)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Pass serializable data to the worker process
            futures = [executor.submit(_solve_pareto_point_worker, objectives, constraints.__dict__, epsilon) for epsilon in epsilon_values]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        pareto_solutions.append(result)
                except Exception as e:
                    logger.error(f"Error in Pareto worker: {e}")
                    
        return pareto_solutions
    
    def _get_objective_bounds(self, objective_name, objective_funcs, variables, constraints):
        """Solves for the min and max of a given objective to set epsilon range."""
        min_opt = Optimize()
        max_opt = Optimize()

        self._add_basic_constraints(min_opt, **variables, constraints=constraints)
        option_expressions_min = self._create_option_pricing_expressions(
            variables['opportunity_cost'], variables['volatility']
        )
        min_opt.add(option_expressions_min['constraints'])

        self._add_basic_constraints(max_opt, **variables, constraints=constraints)
        option_expressions_max = self._create_option_pricing_expressions(
            variables['opportunity_cost'], variables['volatility']
        )
        max_opt.add(option_expressions_max['constraints'])

        min_opt.minimize(objective_funcs[objective_name])
        max_opt.maximize(objective_funcs[objective_name])

        min_val, max_val = None, None
        if min_opt.check() == sat:
            min_val = float(min_opt.model().eval(objective_funcs[objective_name]).as_decimal(9))
        if max_opt.check() == sat:
            max_val = float(max_opt.model().eval(objective_funcs[objective_name]).as_decimal(9))
            
        return min_val, max_val
    
    def _define_variables(self) -> Dict[str, Union[Real, Int]]:
        """Define Z3 variables"""
        return {
            'total_supply': Real('total_supply'),
            'airdrop_percent': Real('airdrop_percent'),
            'launch_price': Real('launch_price'),
            'opportunity_cost': Real('opportunity_cost'),
            'volatility': Real('volatility'),
            'gas_cost': Real('gas_cost'),
            'campaign_duration': Int('campaign_duration'),
            'airdrop_certainty': Real('airdrop_certainty')
        }
    
    def _add_basic_constraints(self, solver, total_supply, airdrop_percent,
                             launch_price, opportunity_cost, volatility,
                             gas_cost, campaign_duration, airdrop_certainty,
                             constraints: SolverConstraints):
        """Add basic constraints to solver"""
        min_supply = constraints.min_supply if constraints.min_supply is not None else 1_000_000
        max_supply = constraints.max_supply if constraints.max_supply is not None else 10_000_000_000
        solver.add(total_supply >= min_supply, total_supply <= max_supply)
        
        solver.add(airdrop_percent >= constraints.min_airdrop_percent, airdrop_percent <= constraints.max_airdrop_percent)
        
        min_price = constraints.min_price if constraints.min_price is not None else 0.01
        max_price = constraints.max_price if constraints.max_price is not None else 10.0
        solver.add(launch_price >= min_price, launch_price <= max_price)
        
        solver.add(opportunity_cost >= 2, opportunity_cost <= 50)
        solver.add(volatility >= 30, volatility <= 150)
        solver.add(gas_cost >= 10, gas_cost <= 500)
        solver.add(campaign_duration >= 3, campaign_duration <= 24)
        solver.add(airdrop_certainty >= 50, airdrop_certainty <= 100)
        
        if constraints.opportunity_cost is not None:
            solver.add(opportunity_cost == constraints.opportunity_cost)
        if constraints.volatility is not None:
            solver.add(volatility == constraints.volatility)
        if constraints.gas_cost is not None:
            solver.add(gas_cost == constraints.gas_cost)
        if constraints.campaign_duration is not None:
            solver.add(campaign_duration == constraints.campaign_duration)
    
    def _create_option_pricing_expressions(self, opportunity_cost: Real, volatility: Real, delta: float = 0.0) -> Dict:
        """
        Creates Z3 expressions for option pricing parameters based on the model
        in "Retroactive Airdrops are Exotic Options".
        
        Uses the correct perpetual American option formula with delta=0.0 for airdrops.
        """
        r = opportunity_cost / 100
        sigma = volatility / 100
        sigma_squared = sigma * sigma

        # Correct formula: a = 0.5 - (r - δ) / σ²
        a = 0.5 - (r - delta) / sigma_squared
        discriminant_val = a * a + 2 * r / sigma_squared

        # Create unique variable name to avoid conflicts
        sqrt_var_name = f"sqrt_disc_{abs(hash((str(opportunity_cost), str(volatility), delta)))}"
        sqrt_discriminant = Real(sqrt_var_name)

        beta = a + sqrt_discriminant
        
        # Robust hurdle rate calculation
        # Hurdle rate = β/(β-1), only defined when β > 1
        hurdle_rate = If(
            beta > 1.01,  # Ensure beta is sufficiently greater than 1
            beta / (beta - 1), 
            10.0  # Maximum hurdle rate when beta is too close to 1
        )

        constraints = [
            sqrt_discriminant * sqrt_discriminant == discriminant_val,
            sqrt_discriminant >= 0,
            beta >= 1.2,  # Ensure beta > 1 with margin for numerical stability
            hurdle_rate >= 1.1,  # Minimum viable hurdle rate (10% return)
            hurdle_rate <= 10.0   # Maximum reasonable hurdle rate (900% return)
        ]

        return {
            "beta": beta,
            "hurdle_rate": hurdle_rate,
            "constraints": constraints
        }
    
    def _extract_solution(self, model, variables: Dict, option_expressions: Optional[Dict] = None, penalties: Optional[Dict] = None) -> AirdropParameters:
        """Extract solution from Z3 model"""
        def _to_float(val):
            return float(val.as_decimal(9).replace('?', ''))

        try:
            solution_dict = {
                'total_supply': _to_float(model.eval(variables['total_supply'], model_completion=True)),
                'airdrop_percent': _to_float(model.eval(variables['airdrop_percent'], model_completion=True)),
                'launch_price': _to_float(model.eval(variables['launch_price'], model_completion=True)),
                'opportunity_cost': _to_float(model.eval(variables['opportunity_cost'], model_completion=True)),
                'volatility': _to_float(model.eval(variables['volatility'], model_completion=True)),
                'gas_cost': _to_float(model.eval(variables['gas_cost'], model_completion=True)),
                'campaign_duration': int(model.eval(variables['campaign_duration'], model_completion=True).as_long()),
                'airdrop_certainty': _to_float(model.eval(variables['airdrop_certainty'], model_completion=True)),
                'revenue_share': 10.0,
                'vesting_months': 18,
                'immediate_unlock': 30.0,
                'penalties': penalties
            }
            if option_expressions:
                solution_dict['beta'] = _to_float(model.eval(option_expressions['beta'], model_completion=True))
                solution_dict['hurdle_rate'] = _to_float(model.eval(option_expressions['hurdle_rate'], model_completion=True))
            else:
                # Compute beta and hurdle_rate numerically from the solution values
                opportunity_cost = solution_dict['opportunity_cost']
                volatility = solution_dict['volatility']
                
                r = opportunity_cost / 100
                sigma = volatility / 100
                sigma_squared = sigma * sigma
                
                # Use consistent delta=0.0 for airdrops (no dividends)
                delta = 0.0
                a = 0.5 - (r - delta) / sigma_squared
                discriminant = a * a + 2 * r / sigma_squared
                
                if discriminant >= 0:
                    beta = a + discriminant**0.5
                    
                    # Check for valid beta (must be > 1 for meaningful hurdle rate)
                    if beta <= 1.0:
                        logger.warning(f"Calculated beta {beta} <= 1.0, using minimum viable value")
                        beta = 1.2
                        hurdle_rate = beta / (beta - 1)  # = 1.2 / 0.2 = 6.0
                    elif beta < 1.01:
                        # Beta too close to 1, would give extreme hurdle rate
                        logger.warning(f"Beta {beta} too close to 1, using maximum hurdle rate")
                        hurdle_rate = 10.0
                    else:
                        hurdle_rate = beta / (beta - 1)
                        # Bound to reasonable range
                        hurdle_rate = max(1.1, min(hurdle_rate, 10.0))
                else:
                    logger.warning(f"Negative discriminant in extraction: {discriminant}")
                    beta = 1.5  # Default fallback for negative discriminant
                    hurdle_rate = 3.0  # beta/(beta-1) = 1.5/0.5 = 3.0
                
                solution_dict['beta'] = beta
                solution_dict['hurdle_rate'] = hurdle_rate
            
            return AirdropParameters(**solution_dict)
        except Exception as e:
            logger.error(f"Error extracting solution: {e}")
            raise SolverError(f"Failed to extract solution: {e}")
    
    def _extract_solution_dict(self, model, variables: Dict) -> Dict:
        """Extract solution as dictionary"""
        return {
            'total_supply': float(model[variables['total_supply']].as_decimal(9)),
            'airdrop_percent': float(model[variables['airdrop_percent']].as_decimal(9)),
            'launch_price': float(model[variables['launch_price']].as_decimal(9)),
            'opportunity_cost': float(model[variables['opportunity_cost']].as_decimal(9)),
            'volatility': float(model[variables['volatility']].as_decimal(9)),
            'gas_cost': float(model[variables['gas_cost']].as_decimal(9)),
            'campaign_duration': int(model[variables['campaign_duration']].as_long()),
            'airdrop_certainty': float(model[variables['airdrop_certainty']].as_decimal(9))
        }
