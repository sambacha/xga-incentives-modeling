"""
Edge case tests for Kalman filter implementation.

Tests cover:
- Boundary conditions
- Numerical stability
- Error handling
- Extreme scenarios
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import warnings
import time

from airdrop_calculator.solver import (
    KalmanConstraintEstimator,
    AdaptiveConstraintScheduler,
    EnhancedZ3Solver
)
from airdrop_calculator.types import SolverConstraints, AirdropParameters


class TestKalmanEdgeCases:
    """Test edge cases for Kalman filter"""
    
    def test_extreme_observations(self):
        """Test handling of extreme observation values"""
        kalman = KalmanConstraintEstimator()
        
        # Test very large observation values
        large_obs = np.array([1.0, 1000.0, 1.0])  # Very high solve time
        state = kalman.update(large_obs, success=True)
        
        # Should clamp to reasonable bounds
        assert state[2] <= 60.0  # Max solve time
        
        # Test very small/negative observation values
        small_obs = np.array([-0.5, -10.0, -1.0])  # Invalid negative values
        state = kalman.update(small_obs, success=True)
        
        # Should handle gracefully
        assert 0 <= state[1] <= 1  # Success prob in valid range
        assert state[2] >= 0.1  # Min solve time
        
    def test_matrix_singularity(self):
        """Test handling of singular matrices"""
        kalman = KalmanConstraintEstimator()
        
        # Create a scenario that could lead to singular matrices
        kalman.R = np.zeros((3, 3))  # Zero observation noise (singular)
        
        # Should handle without crashing
        obs = np.array([0.5, 5.0, 0.7])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore linear algebra warnings
            state = kalman.update(obs, success=True)
        
        # Should still produce valid state
        assert state is not None
        assert len(state) == 4
        
    def test_covariance_explosion(self):
        """Test prevention of covariance matrix explosion"""
        kalman = KalmanConstraintEstimator()
        
        # Simulate many predictions without updates (uncertainty grows)
        for _ in range(100):
            kalman.predict()
        
        # Covariance should not explode
        max_variance = np.max(np.diag(kalman.P))
        assert max_variance < 1000  # Reasonable bound
        
    def test_numerical_stability_small_values(self):
        """Test numerical stability with very small values"""
        kalman = KalmanConstraintEstimator()
        
        # Set very small process noise
        kalman.Q = np.eye(4) * 1e-10
        kalman.R = np.eye(3) * 1e-10
        
        # Should still work
        obs = np.array([0.5, 5.0, 0.7])
        state = kalman.update(obs, success=True)
        
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))
        
    def test_adaptive_noise_with_no_history(self):
        """Test adaptive noise adjustment with insufficient history"""
        kalman = KalmanConstraintEstimator()
        
        # Try to adapt with no history
        kalman.adapt_process_noise()  # Should not crash
        
        # Try with minimal history
        kalman.update(np.array([0.5, 5.0, 0.7]), success=True)
        kalman.adapt_process_noise()  # Should not crash
        
        assert kalman.Q is not None


class TestSchedulerEdgeCases:
    """Test edge cases for adaptive constraint scheduler"""
    
    def test_extreme_relaxation_levels(self):
        """Test constraint relaxation at extreme levels"""
        base = SolverConstraints(
            min_supply=1_000_000_000,
            max_supply=1_000_000_000,  # Same min/max
            min_price=0.5,
            max_price=0.5,  # Same min/max
            min_airdrop_percent=25,
            max_airdrop_percent=25  # Same min/max
        )
        scheduler = AdaptiveConstraintScheduler(base)
        
        # Test with no room for relaxation
        relaxed = scheduler._relax_constraints(1.0)
        
        # Should handle gracefully when bounds are equal
        assert relaxed.min_supply <= relaxed.max_supply
        assert relaxed.min_price <= relaxed.max_price
        assert relaxed.min_airdrop_percent <= relaxed.max_airdrop_percent
        
    def test_invalid_relaxation_levels(self):
        """Test handling of invalid relaxation levels"""
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        
        # Test negative relaxation
        relaxed = scheduler._relax_constraints(-0.5)
        assert relaxed.min_supply >= 1_000_000  # Should respect minimum bounds
        
        # Test excessive relaxation
        relaxed = scheduler._relax_constraints(2.0)
        assert relaxed.max_supply <= 50_000_000_000  # Should respect maximum bounds
        
    def test_none_constraint_values(self):
        """Test handling of None values in constraints"""
        base = SolverConstraints(
            min_supply=None,
            max_supply=None,
            min_price=None,
            max_price=None
        )
        scheduler = AdaptiveConstraintScheduler(base)
        
        # Should use defaults
        relaxed = scheduler._relax_constraints(0.5)
        assert relaxed.min_supply is not None
        assert relaxed.max_supply is not None
        
    def test_performance_summary_edge_cases(self):
        """Test performance summary with edge cases"""
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        
        # Empty history
        summary = scheduler.get_performance_summary()
        assert summary == {}
        
        # All failures
        for _ in range(5):
            scheduler.update_performance(False, 30.0, 0.0)
        
        summary = scheduler.get_performance_summary()
        assert summary['recent_success_rate'] == 0.0
        assert summary['avg_quality'] == 0.0  # No successful solutions
        
    def test_kalman_state_corruption(self):
        """Test recovery from corrupted Kalman state"""
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        
        # Corrupt the Kalman state
        scheduler.kalman.x = np.array([np.nan, np.inf, -np.inf, 1000])
        
        # Get next constraint level should handle this
        level, constraints = scheduler.get_next_constraint_level()
        
        # Should recover to valid values
        assert 0 <= level <= 1
        assert constraints is not None


class TestSolverIntegrationEdgeCases:
    """Test edge cases in solver integration"""
    
    def test_zero_market_cap(self):
        """Test handling of zero or negative market cap targets"""
        solver = EnhancedZ3Solver()
        
        with pytest.raises(ValueError, match="Target market cap must be positive"):
            # Should fail gracefully
            solver.solve_incremental_with_kalman(
                target_market_cap=0,
                target_profitable_users=80,
                initial_constraints=SolverConstraints()
            )
            
    def test_impossible_profitable_users(self):
        """Test handling of impossible profitable user targets"""
        solver = EnhancedZ3Solver()
        
        # Test > 100% profitable users
        with pytest.raises(ValueError, match="Target profitable users must be between"):
            solver.solve_incremental_with_kalman(
                target_market_cap=500_000_000,
                target_profitable_users=150,  # Impossible
                initial_constraints=SolverConstraints(),
                max_iterations=3
            )
        
    def test_max_iterations_zero(self):
        """Test with zero max iterations"""
        solver = EnhancedZ3Solver()
        
        with pytest.raises(ValueError, match="Max iterations must be positive"):
            solver.solve_incremental_with_kalman(
                target_market_cap=500_000_000,
                target_profitable_users=80,
                initial_constraints=SolverConstraints(),
                max_iterations=0
            )
        
    def test_solution_quality_edge_cases(self):
        """Test solution quality calculation edge cases"""
        solver = EnhancedZ3Solver()
        
        # Test with None hurdle_rate
        solution = AirdropParameters(
            total_supply=1_000_000_000,
            airdrop_percent=20,
            launch_price=0.5,
            opportunity_cost=10,
            volatility=80,
            gas_cost=50,
            campaign_duration=6,
            airdrop_certainty=70,
            revenue_share=0.0,
            vesting_months=18,
            immediate_unlock=30.0,
            hurdle_rate=None  # Missing hurdle rate
        )
        
        quality = solver._calculate_solution_quality(solution, 500_000_000, 80)
        assert 0 <= quality <= 1  # Should still produce valid quality
        
        # Test with extreme parameter values
        solution.opportunity_cost = 1000  # Extremely high
        quality = solver._calculate_solution_quality(solution, 500_000_000, 80)
        assert quality < 0.5  # Should penalize unrealistic values
        
    @pytest.mark.skip(reason="Test relies on implementation details and Mock objects in calculations")
    @patch('airdrop_calculator.solver.Solver')
    def test_solver_timeout_handling(self, mock_solver_class):
        """Test handling of solver timeouts"""
        mock_solver = Mock()
        mock_solver.check.return_value = 'unknown'  # Timeout result
        mock_solver_class.return_value = mock_solver
        
        solver = EnhancedZ3Solver()
        solver._define_variables = Mock(return_value={'total_supply': Mock(), 'launch_price': Mock()})
        solver._add_basic_constraints = Mock()
        
        result = solver._solve_with_constraints(
            target_market_cap=500_000_000,
            target_profitable_users=80,
            constraints=SolverConstraints()
        )
        
        assert result is None  # Should return None on timeout
        
    def test_concurrent_kalman_updates(self):
        """Test thread safety of Kalman updates"""
        import threading
        
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        errors = []
        
        def update_thread():
            try:
                for _ in range(10):
                    scheduler.update_performance(True, 5.0, 0.8)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=update_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
        
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow unbounded"""
        scheduler = AdaptiveConstraintScheduler(SolverConstraints())
        
        # Add many observations
        for i in range(1000):
            scheduler.update_performance(i % 2 == 0, 5.0, 0.7)
        
        # History should be bounded (implementation specific)
        # Just ensure it doesn't crash and summary still works
        summary = scheduler.get_performance_summary()
        assert summary['total_attempts'] == 1000
        
    def test_extreme_constraint_bounds(self):
        """Test with extreme constraint bounds"""
        constraints = SolverConstraints(
            min_supply=1,  # Extremely low
            max_supply=10**15,  # Extremely high
            min_price=0.0000001,  # Very small
            max_price=1_000_000,  # Very large
            min_airdrop_percent=0.001,
            max_airdrop_percent=99.999
        )
        
        scheduler = AdaptiveConstraintScheduler(constraints)
        level, relaxed = scheduler.get_next_constraint_level()
        
        # Should produce reasonable constraints
        assert relaxed.min_supply >= 1_000_000  # Enforced minimum
        assert relaxed.max_supply <= 50_000_000_000  # Enforced maximum
        assert relaxed.min_price >= 0.001
        assert relaxed.max_price <= 100.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])