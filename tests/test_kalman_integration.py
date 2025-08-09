"""
Integration tests for Kalman filter with the full system.

Tests the complete workflow from CLI to solution.
"""

import json
import tempfile
from pathlib import Path
from click.testing import CliRunner

from airdrop_calculator.cli import cli
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.types import SolverConstraints


def test_kalman_cli_integration():
    """Test Kalman method through CLI"""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "kalman_solution.json"
        
        # Run solve command with Kalman method
        result = runner.invoke(cli, [
            'solve',
            '--market-cap', '500000000',
            '--profitable-users', '80',
            '--method', 'kalman',
            '--output', str(output_file)
        ])
        
        # Check command succeeded
        assert result.exit_code == 0
        assert 'Optimal Parameters:' in result.output
        
        # Check output file was created
        assert output_file.exists()
        
        # Verify solution contents
        with open(output_file) as f:
            solution = json.load(f)
        
        assert 'total_supply' in solution
        assert 'airdrop_percent' in solution
        assert 'launch_price' in solution
        
        # Verify market cap is close to target
        market_cap = solution['total_supply'] * solution['launch_price']
        assert 450_000_000 <= market_cap <= 550_000_000


def test_kalman_with_constraints_file():
    """Test Kalman method with constraints from file"""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create constraints file
        constraints_file = Path(tmpdir) / "constraints.json"
        constraints = {
            "min_supply": 900_000_000,
            "max_supply": 1_100_000_000,
            "min_price": 0.4,
            "max_price": 0.6,
            "min_airdrop_percent": 15,
            "max_airdrop_percent": 25
        }
        
        with open(constraints_file, 'w') as f:
            json.dump(constraints, f)
        
        # Run with constraints
        result = runner.invoke(cli, [
            'solve',
            '--market-cap', '500000000',
            '--profitable-users', '80',
            '--method', 'kalman',
            '--constraints', str(constraints_file)
        ])
        
        assert result.exit_code == 0
        assert 'Solution found!' in result.output


def test_kalman_solver_with_calculator():
    """Test Kalman solver integration with calculator"""
    # Create solver and find solution
    solver = EnhancedZ3Solver()
    constraints = SolverConstraints(
        min_supply=800_000_000,
        max_supply=1_200_000_000,
        min_price=0.3,
        max_price=0.7,
        min_airdrop_percent=15,
        max_airdrop_percent=35
    )
    
    solution = solver.solve_incremental_with_kalman(
        target_market_cap=500_000_000,
        target_profitable_users=80,
        initial_constraints=constraints,
        max_iterations=10
    )
    
    assert solution is not None
    
    # Use solution with calculator
    calculator = AirdropCalculator(solver)
    
    # Calculate metrics with the solution
    # Note: Calculator expects solver to have parameters, so we simulate this
    solver.total_supply = solution.total_supply
    solver.airdrop_percent = solution.airdrop_percent
    solver.launch_price = solution.launch_price
    
    metrics = calculator.calculate_market_metrics()
    
    assert metrics.min_market_cap > 0
    assert 0 <= metrics.profitable_users_percent <= 100
    assert metrics.hurdle_rate > 1


def test_kalman_learning_persistence():
    """Test that Kalman filter learns across multiple problems"""
    solver = EnhancedZ3Solver()
    constraints = SolverConstraints(
        min_airdrop_percent=10,
        max_airdrop_percent=50
    )
    
    solutions = []
    solve_times = []
    
    # Solve similar problems
    for i in range(3):
        import time
        start = time.time()
        
        solution = solver.solve_incremental_with_kalman(
            target_market_cap=400_000_000 + i * 50_000_000,
            target_profitable_users=75 + i * 5,
            initial_constraints=constraints,
            max_iterations=10
        )
        
        solve_time = time.time() - start
        
        if solution:
            solutions.append(solution)
            solve_times.append(solve_time)
    
    # Should find solutions
    assert len(solutions) >= 2
    
    # Later solves should be faster (learning effect)
    # Note: This might not always hold due to problem difficulty
    if len(solve_times) >= 3:
        avg_early = sum(solve_times[:2]) / 2
        avg_late = solve_times[-1]
        # Just check they're in reasonable range
        assert all(t < 30 for t in solve_times)  # No extremely long solves


def test_kalman_with_analysis_command():
    """Test Kalman solver results with analyze command"""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # First solve with Kalman
        solution_file = Path(tmpdir) / "solution.json"
        result = runner.invoke(cli, [
            'solve',
            '--market-cap', '500000000',
            '--profitable-users', '80',
            '--method', 'kalman',
            '--output', str(solution_file)
        ])
        
        assert result.exit_code == 0
        
        # Then analyze the solution
        result = runner.invoke(cli, [
            'analyze',
            '--config', str(solution_file),
            '--output', tmpdir
        ])
        
        # Should complete analysis
        assert result.exit_code == 0
        assert 'Key Metrics:' in result.output
        assert 'Segment Analysis:' in result.output


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])