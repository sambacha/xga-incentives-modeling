# Airdrop Calculator Documentation

This document provides a guide to using the Airdrop Calculator, a tool for designing and analyzing retroactive airdrop strategies based on the "Retroactive Airdrops are Exotic Options" framework.

## Running Calculations

The Airdrop Calculator offers several methods for running calculations, each suited for different analysis goals.

### 1. Solving with Non-Linear Constraints

The `solve_with_nonlinear_constraints` method is designed to find a set of airdrop parameters that meet specific targets for market capitalization and user profitability. This method is ideal when you have clear, hard targets and want to find a single, optimal solution.

**Example:**

```python
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

solver = EnhancedZ3Solver()
constraints = SolverConstraints(min_airdrop_percent=10, max_airdrop_percent=20)
solution = solver.solve_with_nonlinear_constraints(
    target_market_cap=200_000_000,
    target_profitable_users=60,
    constraints=constraints
)

if solution:
    print("Solution found:", solution)
else:
    print("No solution found.")
```

### 2. Solving with Soft Constraints

The `solve_with_soft_constraints` method provides a more flexible approach by allowing you to define objectives with different weights. This is useful when you want to balance multiple competing goals, such as maximizing market cap while minimizing the airdrop percentage.

**Example:**

```python
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

solver = EnhancedZ3Solver()
constraints = SolverConstraints()
objectives = {
    'market_cap': {'target': 250_000_000, 'weight': 1.0},
    'profitable_users': {'target': 70, 'weight': 0.5},
}
solution = solver.solve_with_soft_constraints(objectives, constraints)

if solution:
    print("Solution with soft constraints:", solution)
```

### 3. Incremental Solving with Relaxation

The `solve_incremental_with_relaxation` method is designed for situations where the initial constraints may be too strict. It allows you to define multiple levels of constraints, and the solver will attempt to find a solution by incrementally relaxing them.

**Example:**

```python
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

solver = EnhancedZ3Solver()
constraint_levels = [
    (100, SolverConstraints(min_price=5.0)),
    (50, SolverConstraints(min_price=1.0))
]
solution = solver.solve_incremental_with_relaxation(
    target_market_cap=200_000_000,
    target_profitable_users=60,
    constraint_levels=constraint_levels
)

if solution:
    print("Solution with relaxed constraints:", solution)
```

### 4. Finding Pareto-Optimal Solutions

The `find_pareto_optimal_solutions` method is used to explore the trade-offs between two competing objectives. It returns a set of solutions that are optimal in the sense that you cannot improve one objective without worsening the other. This is useful for understanding the solution space and making informed decisions.

**Example:**

```python
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints

solver = EnhancedZ3Solver()
constraints = SolverConstraints()
solutions = solver.find_pareto_optimal_solutions(
    objectives=['market_cap', 'profitability'],
    constraints=constraints,
    num_solutions=5
)

for sol in solutions:
    print("Pareto-optimal solution:", sol)
```

## Designing Scenarios

The `AirdropCalculator` also allows you to define and run scenarios to model different airdrop strategies.

### Defining a Scenario

A scenario is defined using the `Scenario` dataclass, which includes a name, description, constraints, and objectives.

```python
from airdrop_calculator.types import Scenario

scenario = Scenario(
    name="Aggressive Growth",
    description="A scenario focused on maximizing market cap.",
    constraints={
        "min_airdrop_percent": 15,
        "max_airdrop_percent": 30,
    },
    objectives={
        "market_cap": {"target": 500_000_000, "weight": 1.5},
        "profitability": {"target": 50, "weight": 0.5},
    }
)
```

### Running and Comparing Scenarios

The `AirdropCalculator` can run a single scenario or compare multiple scenarios to help you choose the best strategy.

```python
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver

solver = EnhancedZ3Solver()
calculator = AirdropCalculator(solver)

# Run a single scenario
result = calculator.run_scenario(scenario)
print("Scenario result:", result)

# Compare multiple scenarios
scenario2 = Scenario(...)
comparison = calculator.compare_scenarios([scenario, scenario2])
print("Comparison result:", comparison)
```

By using these methods, you can design and analyze a wide range of airdrop strategies to achieve your desired outcomes.
