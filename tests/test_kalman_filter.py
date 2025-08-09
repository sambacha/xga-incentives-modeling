"""
Tests for Kalman filter-enhanced incremental solver.

Tests cover:
- KalmanConstraintEstimator functionality
- AdaptiveConstraintScheduler behavior
- Integration with EnhancedZ3Solver
- Performance and learning capabilities
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from airdrop_calculator.solver import (
    KalmanConstraintEstimator, 
    AdaptiveConstraintScheduler,
    EnhancedZ3Solver,
    ConstraintState
)
from airdrop_calculator.types import SolverConstraints, AirdropParameters


class TestKalmanConstraintEstimator:
    """Test the Kalman filter implementation"""
    
    def test_initialization(self):
        """Test Kalman filter initialization"""
        kalman = KalmanConstraintEstimator()
        
        # Check initial state
        assert kalman.dim_state == 4
        assert kalman.dim_obs == 3
        np.testing.assert_array_equal(kalman.x, [0.5, 0.5, 10.0, 0.5])
        
        # Check matrix dimensions
        assert kalman.P.shape == (4, 4)
        assert kalman.Q.shape == (4, 4)
        assert kalman.R.shape == (3, 3)
        assert kalman.F.shape == (4, 4)
        assert kalman.H.shape == (3, 4)
        
    def test_predict(self):
        """Test state prediction"""
        kalman = KalmanConstraintEstimator()
        initial_state = kalman.x.copy()
        
        # Perform prediction
        predicted_state = kalman.predict()
        
        # State should evolve according to F matrix
        expected = kalman.F @ initial_state
        np.testing.assert_array_almost_equal(predicted_state, expected)
        
        # Covariance should increase (uncertainty grows)
        assert np.trace(kalman.P) > np.trace(np.eye(4) * 0.1)
        
    def test_update_success(self):
        """Test update with successful observation"""
        kalman = KalmanConstraintEstimator()
        
        # Successful observation
        observation = np.array([0.8, 5.0, 0.9])  # success prob, time, quality
        updated_state = kalman.update(observation, success=True)
        
        # State should be updated
        assert len(kalman.history) == 1
        assert kalman.history[0]['success'] == True
        
        # Values should be within bounds
        assert 0 <= updated_state[0] <= 1  # relaxation level
        assert 0 <= updated_state[1] <= 1  # success probability
        assert 0.1 <= updated_state[2] <= 60  # solve time
        assert 0 <= updated_state[3] <= 1  # quality
        
    def test_update_failure(self):
        """Test update with failed observation"""
        kalman = KalmanConstraintEstimator()
        
        # Failed observation (high time, but marked as failure)
        observation = np.array([0.0, 50.0, 0.0])
        updated_state = kalman.update(observation, success=False)
        
        # Should handle failure appropriately
        assert kalman.history[0]['success'] == False
        assert kalman.history[0]['observation'][0] == 0.0  # success = 0
        assert kalman.history[0]['observation'][2] == 0.1  # low quality
        
    def test_adaptive_process_noise(self):
        """Test adaptive process noise adjustment"""
        kalman = KalmanConstraintEstimator()
        
        # Add varied observations
        observations = [
            (np.array([0.7, 3.0, 0.8]), True),
            (np.array([0.6, 8.0, 0.7]), True),
            (np.array([0.9, 2.0, 0.95]), True),
            (np.array([0.4, 15.0, 0.6]), True),
            (np.array([0.8, 4.0, 0.85]), True),
        ]
        
        initial_Q = kalman.Q.copy()
        
        for obs, success in observations:
            kalman.update(obs, success)
        
        kalman.adapt_process_noise()
        
        # Process noise should have adapted
        assert not np.array_equal(kalman.Q, initial_Q)
        
    def test_getters(self):
        """Test getter methods"""
        kalman = KalmanConstraintEstimator()
        kalman.x = np.array([0.7, 0.85, 5.5, 0.92])
        
        assert kalman.get_optimal_relaxation() == 0.7
        assert kalman.get_success_probability() == 0.85
        assert kalman.get_expected_time() == 5.5


class TestAdaptiveConstraintScheduler:
    """Test the adaptive constraint scheduler"""
    
    def test_initialization(self):
        """Test scheduler initialization"""
        base_constraints = SolverConstraints(
            min_airdrop_percent=10,
            max_airdrop_percent=30
        )
        scheduler = AdaptiveConstraintScheduler(base_constraints)
        
        assert scheduler.base_constraints == base_constraints
        assert isinstance(scheduler.kalman, KalmanConstraintEstimator)
        assert scheduler.relaxation_history == []
        
    def test_constraint_relaxation(self):
        """Test constraint relaxation at different levels"""
        base_constraints = SolverConstraints(
            min_supply=1_000_000_000,
            max_supply=2_000_000_000,
            min_price=0.5,
            max_price=1.0,
            min_airdrop_percent=20,
            max_airdrop_percent=40
        )
        scheduler = AdaptiveConstraintScheduler(base_constraints)
        
        # Test no relaxation (level=0)
        relaxed = scheduler._relax_constraints(0.0)
        assert relaxed.min_supply == base_constraints.min_supply
        assert relaxed.max_supply == base_constraints.max_supply
        
        # Test partial relaxation (level=0.5)
        relaxed = scheduler._relax_constraints(0.5)
        assert relaxed.min_supply < base_constraints.min_supply
        assert relaxed.max_supply > base_constraints.max_supply
        assert relaxed.min_airdrop_percent < base_constraints.min_airdrop_percent
        assert relaxed.max_airdrop_percent > base_constraints.max_airdrop_percent
        
        # Test full relaxation (level=1.0)
        relaxed = scheduler._relax_constraints(1.0)
        assert relaxed.min_supply < base_constraints.min_supply
        assert relaxed.max_supply > base_constraints.max_supply
        
        # Check bounds are respected
        assert relaxed.min_supply >= 1_000_000
        assert relaxed.max_supply <= 50_000_000_000
        assert relaxed.min_price >= 0.001
        assert relaxed.max_price <= 100.0
        
    def test_get_next_constraint_level(self):
        """Test getting next constraint level"""
        base_constraints = SolverConstraints(
            min_airdrop_percent=15,
            max_airdrop_percent=35
        )
        scheduler = AdaptiveConstraintScheduler(base_constraints)
        
        # Get next level
        level, constraints = scheduler.get_next_constraint_level()
        
        assert 0 <= level <= 1
        assert isinstance(constraints, SolverConstraints)
        
    def test_update_performance(self):
        """Test performance update tracking"""
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        
        # Update with success
        scheduler.update_performance(success=True, solve_time=2.5, solution_quality=0.85)
        
        assert len(scheduler.relaxation_history) == 1
        assert scheduler.relaxation_history[0]['success'] == True
        assert scheduler.relaxation_history[0]['solve_time'] == 2.5
        assert scheduler.relaxation_history[0]['quality'] == 0.85
        
        # Update with failure
        scheduler.update_performance(success=False, solve_time=30.0, solution_quality=0.0)
        
        assert len(scheduler.relaxation_history) == 2
        assert scheduler.relaxation_history[1]['success'] == False
        
    def test_performance_summary(self):
        """Test performance summary generation"""
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        
        # Empty summary
        summary = scheduler.get_performance_summary()
        assert summary == {}
        
        # Add some performance data
        for i in range(12):
            success = i % 3 != 0  # 2/3 success rate
            scheduler.update_performance(
                success=success,
                solve_time=5.0 + i,
                solution_quality=0.7 if success else 0.0
            )
        
        summary = scheduler.get_performance_summary()
        
        assert summary['total_attempts'] == 12
        assert 0.6 <= summary['recent_success_rate'] <= 0.8  # Around 2/3
        assert summary['avg_solve_time'] > 5.0
        assert 0 <= summary['current_relaxation'] <= 1
        assert 0 <= summary['predicted_success_prob'] <= 1


class TestKalmanIntegration:
    """Test integration with EnhancedZ3Solver"""
    
    @patch('airdrop_calculator.solver.Solver')
    def test_solve_with_constraints_helper(self, mock_solver_class):
        """Test the _solve_with_constraints helper method"""
        # Setup mock solver
        mock_solver = Mock()
        mock_model = Mock()
        mock_solver.check.return_value = 'sat'
        mock_solver.model.return_value = mock_model
        mock_solver_class.return_value = mock_solver
        
        solver = EnhancedZ3Solver()
        
        # Mock the extraction method
        expected_solution = AirdropParameters(
            total_supply=1_000_000_000,
            airdrop_percent=20,
            launch_price=0.5,
            opportunity_cost=10,
            volatility=80,
            gas_cost=50,
            campaign_duration=6,
            airdrop_certainty=70
        )
        solver._extract_solution = Mock(return_value=expected_solution)
        solver._define_variables = Mock(return_value={
            'total_supply': Mock(),
            'launch_price': Mock()
        })
        solver._add_basic_constraints = Mock()
        
        # Test solving
        solution = solver._solve_with_constraints(
            target_market_cap=500_000_000,
            target_profitable_users=80,
            constraints=SolverConstraints(),
            expected_time=5.0
        )
        
        assert solution == expected_solution
        mock_solver.set.assert_called_once_with("timeout", 5000)
        
    def test_calculate_solution_quality(self):
        """Test solution quality calculation"""
        solver = EnhancedZ3Solver()
        
        # Perfect solution
        solution = AirdropParameters(
            total_supply=1_000_000_000,
            airdrop_percent=20,
            launch_price=0.5,  # Gives 500M market cap
            opportunity_cost=10,
            volatility=80,
            gas_cost=50,
            campaign_duration=6,
            airdrop_certainty=70,
            hurdle_rate=2.5
        )
        
        quality = solver._calculate_solution_quality(solution, 500_000_000, 80)
        assert quality > 0.9  # Should be high quality
        
        # Poor solution (far from target)
        solution.launch_price = 0.1  # Gives 100M market cap
        quality = solver._calculate_solution_quality(solution, 500_000_000, 80)
        assert quality < 0.5  # Should be low quality
        
    @patch.object(EnhancedZ3Solver, '_solve_with_constraints')
    def test_solve_incremental_with_kalman_success(self, mock_solve):
        """Test successful Kalman-enhanced solving"""
        solver = EnhancedZ3Solver()
        
        # Mock successful solutions with increasing quality
        solutions = [
            AirdropParameters(
                total_supply=1_000_000_000, airdrop_percent=20,
                launch_price=0.4, opportunity_cost=10, volatility=80,
                gas_cost=50, campaign_duration=6, airdrop_certainty=70
            ),
            AirdropParameters(
                total_supply=1_000_000_000, airdrop_percent=20,
                launch_price=0.5, opportunity_cost=10, volatility=80,
                gas_cost=50, campaign_duration=6, airdrop_certainty=70,
                hurdle_rate=2.5
            )
        ]
        mock_solve.side_effect = solutions
        
        result = solver.solve_incremental_with_kalman(
            target_market_cap=500_000_000,
            target_profitable_users=80,
            initial_constraints=SolverConstraints(),
            max_iterations=5
        )
        
        assert result is not None
        assert result.launch_price == 0.5  # Should return the better solution
        
    @patch.object(EnhancedZ3Solver, '_solve_with_constraints')
    def test_solve_incremental_with_kalman_no_solution(self, mock_solve):
        """Test Kalman solver when no solution is found"""
        solver = EnhancedZ3Solver()
        mock_solve.return_value = None  # No solution found
        
        result = solver.solve_incremental_with_kalman(
            target_market_cap=500_000_000,
            target_profitable_users=80,
            initial_constraints=SolverConstraints(),
            max_iterations=3
        )
        
        assert result is None
        assert mock_solve.call_count == 3  # Should try max_iterations times
        
    @patch.object(EnhancedZ3Solver, '_solve_with_constraints')
    def test_kalman_early_termination(self, mock_solve):
        """Test early termination on high-quality solution"""
        solver = EnhancedZ3Solver()
        
        # High quality solution
        high_quality_solution = AirdropParameters(
            total_supply=1_000_000_000, airdrop_percent=20,
            launch_price=0.5, opportunity_cost=10, volatility=80,
            gas_cost=50, campaign_duration=6, airdrop_certainty=70,
            hurdle_rate=2.5
        )
        mock_solve.return_value = high_quality_solution
        
        # Mock quality calculation to return > 0.9
        solver._calculate_solution_quality = Mock(return_value=0.95)
        
        result = solver.solve_incremental_with_kalman(
            target_market_cap=500_000_000,
            target_profitable_users=80,
            initial_constraints=SolverConstraints(),
            max_iterations=10
        )
        
        assert result == high_quality_solution
        # Should terminate early, not run all 10 iterations
        assert mock_solve.call_count < 10


class TestKalmanPerformance:
    """Test performance characteristics of Kalman filter approach"""
    
    def test_learning_over_iterations(self):
        """Test that Kalman filter learns and improves predictions"""
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        
        # Simulate a pattern: higher relaxation = higher success
        for i in range(20):
            level, _ = scheduler.get_next_constraint_level()
            
            # Simulate success correlation with relaxation level
            success = np.random.random() < (0.3 + 0.6 * level)
            solve_time = 2.0 + 10.0 * (1 - level) + np.random.normal(0, 1)
            quality = 0.5 + 0.4 * level if success else 0.1
            
            scheduler.update_performance(success, solve_time, quality)
        
        # After learning, should prefer higher relaxation levels
        final_summary = scheduler.get_performance_summary()
        assert final_summary['current_relaxation'] > 0.5
        
    def test_convergence_behavior(self):
        """Test that Kalman filter converges to stable estimates"""
        kalman = KalmanConstraintEstimator()
        
        # Feed consistent observations
        true_state = np.array([0.7, 0.8, 5.0, 0.85])
        observation_noise = 0.05
        
        states = []
        for _ in range(50):
            # Add small noise to true observations
            obs = true_state[1:] + np.random.normal(0, observation_noise, 3)
            kalman.update(obs, success=True)
            states.append(kalman.x.copy())
        
        # Check convergence
        final_state = states[-1]
        assert abs(final_state[1] - true_state[1]) < 0.1  # Success prob converged
        assert abs(final_state[2] - true_state[2]) < 1.0  # Time converged
        assert abs(final_state[3] - true_state[3]) < 0.1  # Quality converged
        
        # Variance should decrease over time
        early_variance = np.var(states[:10], axis=0)
        late_variance = np.var(states[-10:], axis=0)
        assert np.mean(late_variance) < np.mean(early_variance)


@pytest.mark.integration
class TestKalmanCLIIntegration:
    """Test CLI integration with Kalman solver"""
    
    def test_cli_kalman_method(self):
        """Test that CLI properly calls Kalman method"""
        from airdrop_calculator.cli import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'solve',
            '--market-cap', '500000000',
            '--profitable-users', '80',
            '--method', 'kalman'
        ])
        
        # Should complete successfully
        assert result.exit_code == 0
        assert 'Optimal Parameters:' in result.output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])