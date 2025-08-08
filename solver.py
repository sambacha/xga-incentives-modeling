
from z3 import *
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class EnhancedZ3Solver:
    """Enhanced Z3 solver with advanced features"""
    
    def __init__(self):
        self.solver = Solver()
        self.optimizer = Optimize()
        self._solution_cache = {}
    
    def solve_with_nonlinear_constraints(self, 
                                        target_market_cap: float,
                                        target_profitable_users: float,
                                        constraints: SolverConstraints) -> Optional[AirdropParameters]:
        """
        Solve using exact non-linear option pricing formulas
        
        But wait, maybe it's better... to use Z3's tactics for 
        better performance on non-linear problems
        """
        # Use specialized solver for non-linear arithmetic
        solver = SolverFor("NRA")  # Non-linear Real Arithmetic
        
        # Define variables
        total_supply = Real('total_supply')
        airdrop_percent = Real('airdrop_percent')
        launch_price = Real('launch_price')
        opportunity_cost = Real('opportunity_cost')
        volatility = Real('volatility')
        gas_cost = Real('gas_cost')
        campaign_duration = Int('campaign_duration')
        airdrop_certainty = Real('airdrop_certainty')
        
        # Add basic constraints
        self._add_basic_constraints(solver, total_supply, airdrop_percent, 
                                  launch_price, opportunity_cost, volatility,
                                  gas_cost, campaign_duration, airdrop_certainty,
                                  constraints)
        
        # Exact beta calculation
        r = opportunity_cost / 100
        sigma = volatility / 100
        sigma_squared = sigma * sigma
        a = RealVal(0.5) - r / sigma_squared
        
        # Use Z3's sqrt function
        discriminant = a * a + 2 * r / sigma_squared
        beta = a + Sqrt(discriminant)
        
        # Hurdle rate constraint
        hurdle_rate = beta / (beta - 1)
        solver.add(hurdle_rate > 1.0)
        solver.add(hurdle_rate < 5.0)  # Reasonable upper bound
        
        # Market cap constraint
        market_cap = total_supply * launch_price
        solver.add(market_cap >= target_market_cap * 0.95)
        solver.add(market_cap <= target_market_cap * 1.05)
        
        # Profitability constraint (simplified for Z3)
        # Lower opportunity cost and higher airdrop percent increase profitability
        profitability_proxy = (30 - opportunity_cost) * 2 + airdrop_percent
        solver.add(profitability_proxy >= target_profitable_users)
        
        # Set timeout for complex solving
        solver.set("timeout", 30000)
        
        if solver.check() == sat:
            model = solver.model()
            return self._extract_solution(model, {
                'total_supply': total_supply,
                'airdrop_percent': airdrop_percent,
                'launch_price': launch_price,
                'opportunity_cost': opportunity_cost,
                'volatility': volatility,
                'gas_cost': gas_cost,
                'campaign_duration': campaign_duration,
                'airdrop_certainty': airdrop_certainty
            })
        
        return None
    
    def solve_with_soft_constraints(self, 
                                  objectives: Dict[str, Tuple[float, float]],
                                  constraints: SolverConstraints) -> Optional[AirdropParameters]:
        """
        Solve with soft constraints and weighted objectives
        objectives: Dict of {param_name: (target_value, weight)}
        """
        opt = Optimize()
        
        # Define variables
        variables = self._define_variables()
        
        # Add hard constraints
        self._add_basic_constraints(opt, **variables, constraints=constraints)
        
        # Create soft constraints with penalties
        penalties = []
        
        for param_name, (target, weight) in objectives.items():
            if param_name == 'market_cap':
                market_cap = variables['total_supply'] * variables['launch_price']
                diff = If(market_cap > target,
                         market_cap - target,
                         target - market_cap)
                penalties.append(weight * diff / target)  # Normalize by target
            
            elif param_name == 'profitable_users':
                # Proxy for profitable users
                profitability = (30 - variables['opportunity_cost']) * 2 + variables['airdrop_percent']
                diff = If(profitability > target,
                         profitability - target,
                         target - profitability)
                penalties.append(weight * diff / 100)
            
            elif param_name in variables:
                var = variables[param_name]
                diff = If(var > target, var - target, target - var)
                penalties.append(weight * diff / target)
        
        # Minimize weighted sum of penalties
        total_penalty = Sum(penalties)
        opt.minimize(total_penalty)
        
        if opt.check() == sat:
            model = opt.model()
            return self._extract_solution(model, variables)
        
        return None
    
    def solve_incremental_with_relaxation(self,
                                        target_market_cap: float,
                                        target_profitable_users: float,
                                        constraint_levels: List[Tuple[int, SolverConstraints]]) -> Optional[AirdropParameters]:
        """
        Incrementally relax constraints if no solution found
        constraint_levels: List of (priority, constraints) tuples
        """
        # Sort by priority (higher = more important)
        sorted_levels = sorted(constraint_levels, key=lambda x: x[0], reverse=True)
        
        for i in range(len(sorted_levels)):
            solver = Solver()
            variables = self._define_variables()
            
            # Add constraints up to current level
            for j in range(i + 1):
                _, constraints = sorted_levels[j]
                self._add_basic_constraints(solver, **variables, constraints=constraints)
            
            # Add target constraints
            market_cap = variables['total_supply'] * variables['launch_price']
            solver.add(And(market_cap >= target_market_cap * 0.9,
                          market_cap <= target_market_cap * 1.1))
            
            if solver.check() == sat:
                logger.info(f"Solution found at constraint level {sorted_levels[i][0]}")
                model = solver.model()
                return self._extract_solution(model, variables)
        
        logger.warning("No solution found even after relaxing all constraints")
        return None
    
    def find_pareto_optimal_solutions(self,
                                    objectives: List[str],
                                    constraints: SolverConstraints,
                                    num_solutions: int = 20) -> List[Dict]:
        """
        Find Pareto-optimal solutions for multiple objectives
        
        But wait, maybe it's better... to use epsilon-constraint method
        for more controlled Pareto frontier generation
        """
        pareto_solutions = []
        
        # Define objective functions
        variables = self._define_variables()
        
        objective_funcs = {
            'market_cap': variables['total_supply'] * variables['launch_price'],
            'total_airdrop_value': variables['total_supply'] * variables['airdrop_percent'] / 100 * variables['launch_price'],
            'hurdle_rate': self._calculate_hurdle_rate_z3(variables['opportunity_cost'], variables['volatility']),
            'profitability': (30 - variables['opportunity_cost']) * 2 + variables['airdrop_percent']
        }
        
        # Generate Pareto points using weighted sum method
        for i in range(num_solutions):
            opt = Optimize()
            
            # Add constraints
            self._add_basic_constraints(opt, **variables, constraints=constraints)
            
            # Exclude previously found solutions
            for sol in pareto_solutions:
                opt.add(Or(
                    variables['total_supply'] != sol['total_supply'],
                    variables['airdrop_percent'] != sol['airdrop_percent'],
                    variables['launch_price'] != sol['launch_price']
                ))
            
            # Random weights for objectives
            weights = np.random.dirichlet(np.ones(len(objectives)))
            
            # Create weighted objective
            weighted_obj = Sum([
                weights[j] * objective_funcs[obj]
                for j, obj in enumerate(objectives)
                if obj in objective_funcs
            ])
            
            # Randomly minimize or maximize
            if np.random.random() > 0.5:
                opt.minimize(weighted_obj)
            else:
                opt.maximize(weighted_obj)
            
            if opt.check() == sat:
                model = opt.model()
                solution = self._extract_solution_dict(model, variables)
                
                # Calculate objective values
                for obj in objectives:
                    if obj in objective_funcs:
                        solution[f'{obj}_value'] = float(model.eval(objective_funcs[obj]).as_decimal(9))
                
                pareto_solutions.append(solution)
        
        # Filter dominated solutions
        return self._filter_dominated_solutions(pareto_solutions, objectives)
    
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
        # Supply constraints
        solver.add(total_supply >= constraints.min_supply or 1_000_000)
        solver.add(total_supply <= constraints.max_supply or 10_000_000_000)
        
        # Airdrop percent constraints
        solver.add(airdrop_percent >= constraints.min_airdrop_percent)
        solver.add(airdrop_percent <= constraints.max_airdrop_percent)
        
        # Price constraints
        solver.add(launch_price >= constraints.min_price or 0.01)
        solver.add(launch_price <= constraints.max_price or 10.0)
        
        # Other constraints
        solver.add(opportunity_cost >= 2)
        solver.add(opportunity_cost <= 50)
        solver.add(volatility >= 30)
        solver.add(volatility <= 150)
        solver.add(gas_cost >= 10)
        solver.add(gas_cost <= 500)
        solver.add(campaign_duration >= 3)
        solver.add(campaign_duration <= 24)
        solver.add(airdrop_certainty >= 50)
        solver.add(airdrop_certainty <= 100)
        
        # Fixed constraints if specified
        if constraints.opportunity_cost is not None:
            solver.add(opportunity_cost == constraints.opportunity_cost)
        if constraints.volatility is not None:
            solver.add(volatility == constraints.volatility)
        if constraints.gas_cost is not None:
            solver.add(gas_cost == constraints.gas_cost)
        if constraints.campaign_duration is not None:
            solver.add(campaign_duration == constraints.campaign_duration)
    
    def _calculate_hurdle_rate_z3(self, opportunity_cost, volatility):
        """Calculate hurdle rate using Z3 expressions"""
        r = opportunity_cost / 100
        sigma = volatility / 100
        sigma_squared = sigma * sigma
        a = RealVal(0.5) - r / sigma_squared
        
        discriminant = a * a + 2 * r / sigma_squared
        beta = a + Sqrt(discriminant)
        
        return beta / (beta - 1)
    
    def _extract_solution(self, model, variables: Dict) -> AirdropParameters:
        """Extract solution from Z3 model"""
        try:
            return AirdropParameters(
                total_supply=float(model[variables['total_supply']].as_decimal(9)),
                airdrop_percent=float(model[variables['airdrop_percent']].as_decimal(9)),
                launch_price=float(model[variables['launch_price']].as_decimal(9)),
                opportunity_cost=float(model[variables['opportunity_cost']].as_decimal(9)),
                volatility=float(model[variables['volatility']].as_decimal(9)),
                gas_cost=float(model[variables['gas_cost']].as_decimal(9)),
                campaign_duration=int(model[variables['campaign_duration']].as_long()),
                airdrop_certainty=float(model[variables['airdrop_certainty']].as_decimal(9)),
                revenue_share=10.0,
                vesting_months=18,
                immediate_unlock=30.0
            )
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
    
    def _filter_dominated_solutions(self, solutions: List[Dict], objectives: List[str]) -> List[Dict]:
        """Filter out dominated solutions to keep only Pareto optimal ones"""
        pareto_optimal = []
        
        for i, sol1 in enumerate(solutions):
            dominated = False
            
            for j, sol2 in enumerate(solutions):
                if i == j:
                    continue
                
                # Check if sol1 is dominated by sol2
                better_count = 0
                equal_count = 0
                
                for obj in objectives:
                    val1 = sol1.get(f'{obj}_value', 0)
                    val2 = sol2.get(f'{obj}_value', 0)
                    
                    if val2 > val1:
                        better_count += 1
                    elif val2 == val1:
                        equal_count += 1
                
                if better_count > 0 and better_count + equal_count == len(objectives):
                    dominated = True
                    break
            
            if not dominated:
                pareto_optimal.append(sol1)
        
        return pareto_optimal