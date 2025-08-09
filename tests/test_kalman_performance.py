"""
Performance comparison tests for Kalman filter vs traditional approaches.

These tests measure:
- Convergence speed
- Solution quality
- Resource efficiency
- Learning capabilities
"""

import pytest
import time
import numpy as np
from typing import List, Dict, Tuple
import logging

from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.types import SolverConstraints, AirdropParameters


# Set up logging for performance analysis
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)


class PerformanceMetrics:
    """Track performance metrics for comparison"""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.solve_times: List[float] = []
        self.iterations: List[int] = []
        self.qualities: List[float] = []
        self.successes: List[bool] = []
        
    def record_solve(self, solve_time: float, iterations: int, 
                    quality: float, success: bool):
        """Record a solve attempt"""
        self.solve_times.append(solve_time)
        self.iterations.append(iterations)
        self.qualities.append(quality)
        self.successes.append(success)
        
    def get_summary(self) -> Dict:
        """Get performance summary statistics"""
        return {
            'method': self.method_name,
            'total_attempts': len(self.solve_times),
            'success_rate': sum(self.successes) / len(self.successes) if self.successes else 0,
            'avg_solve_time': np.mean(self.solve_times) if self.solve_times else 0,
            'avg_iterations': np.mean(self.iterations) if self.iterations else 0,
            'avg_quality': np.mean([q for q, s in zip(self.qualities, self.successes) if s]) if any(self.successes) else 0,
            'best_quality': max(self.qualities) if self.qualities else 0,
            'convergence_rate': self._calculate_convergence_rate()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate how quickly method converges to good solutions"""
        if not self.successes:
            return 0.0
            
        # Find first success
        first_success_idx = next((i for i, s in enumerate(self.successes) if s), None)
        if first_success_idx is None:
            return 0.0
            
        # Find first high-quality success (>0.8)
        high_quality_idx = next((i for i, (q, s) in enumerate(zip(self.qualities, self.successes)) 
                               if s and q > 0.8), None)
        if high_quality_idx is None:
            return 1.0 / (first_success_idx + 1)
            
        return 1.0 / (high_quality_idx + 1)


class TestKalmanPerformanceComparison:
    """Compare Kalman filter performance against traditional methods"""
    
    def _generate_test_scenarios(self) -> List[Tuple[float, float, SolverConstraints]]:
        """Generate diverse test scenarios"""
        scenarios = []
        
        # Easy scenarios
        scenarios.append((
            100_000_000, 70,  # $100M, 70% profitable
            SolverConstraints(
                min_supply=500_000_000, max_supply=2_000_000_000,
                min_price=0.05, max_price=0.5,
                min_airdrop_percent=5, max_airdrop_percent=40
            )
        ))
        
        # Medium scenarios
        scenarios.append((
            500_000_000, 80,  # $500M, 80% profitable
            SolverConstraints(
                min_supply=800_000_000, max_supply=1_200_000_000,
                min_price=0.3, max_price=0.7,
                min_airdrop_percent=15, max_airdrop_percent=35
            )
        ))
        
        # Hard scenarios
        scenarios.append((
            1_000_000_000, 85,  # $1B, 85% profitable
            SolverConstraints(
                min_supply=900_000_000, max_supply=1_100_000_000,
                min_price=0.8, max_price=1.2,
                min_airdrop_percent=20, max_airdrop_percent=30
            )
        ))
        
        # Very constrained scenarios
        scenarios.append((
            750_000_000, 90,  # $750M, 90% profitable
            SolverConstraints(
                min_supply=1_000_000_000, max_supply=1_000_000_000,  # Fixed supply
                min_price=0.7, max_price=0.8,
                min_airdrop_percent=25, max_airdrop_percent=25  # Fixed airdrop
            )
        ))
        
        return scenarios
    
    def test_convergence_speed_comparison(self):
        """Compare how quickly different methods converge to solutions"""
        solver = EnhancedZ3Solver()
        scenarios = self._generate_test_scenarios()
        
        # Track metrics for each method
        kalman_metrics = PerformanceMetrics("Kalman")
        traditional_metrics = PerformanceMetrics("Traditional")
        
        for i, (market_cap, profitable_users, constraints) in enumerate(scenarios):
            perf_logger.info(f"\nScenario {i+1}: ${market_cap/1e6:.0f}M, {profitable_users}% profitable")
            
            # Test Kalman approach
            start_time = time.time()
            iterations_kalman = 0
            
            try:
                # Mock iteration counting
                original_solve = solver._solve_with_constraints
                def count_iterations(*args, **kwargs):
                    nonlocal iterations_kalman
                    iterations_kalman += 1
                    return original_solve(*args, **kwargs)
                
                solver._solve_with_constraints = count_iterations
                
                solution = solver.solve_incremental_with_kalman(
                    market_cap, profitable_users, constraints, max_iterations=10
                )
                
                solve_time = time.time() - start_time
                success = solution is not None
                quality = solver._calculate_solution_quality(solution, market_cap, profitable_users) if success else 0
                
                kalman_metrics.record_solve(solve_time, iterations_kalman, quality, success)
                
            finally:
                solver._solve_with_constraints = original_solve
            
            # Test traditional approach
            start_time = time.time()
            iterations_traditional = 3  # Fixed levels
            
            try:
                levels = [
                    (100, constraints),
                    (50, SolverConstraints(min_airdrop_percent=10, max_airdrop_percent=40)),
                    (10, SolverConstraints())
                ]
                solution = solver.solve_incremental_with_relaxation(
                    market_cap, profitable_users, levels
                )
                
                solve_time = time.time() - start_time
                success = solution is not None
                quality = solver._calculate_solution_quality(solution, market_cap, profitable_users) if success else 0
                
                traditional_metrics.record_solve(solve_time, iterations_traditional, quality, success)
                
            except Exception:
                traditional_metrics.record_solve(30.0, iterations_traditional, 0, False)
        
        # Compare results
        kalman_summary = kalman_metrics.get_summary()
        traditional_summary = traditional_metrics.get_summary()
        
        perf_logger.info("\n" + "="*60)
        perf_logger.info("PERFORMANCE COMPARISON RESULTS")
        perf_logger.info("="*60)
        perf_logger.info(f"\nKalman Filter Method:")
        for key, value in kalman_summary.items():
            perf_logger.info(f"  {key}: {value}")
            
        perf_logger.info(f"\nTraditional Method:")
        for key, value in traditional_summary.items():
            perf_logger.info(f"  {key}: {value}")
        
        # Assertions
        assert kalman_summary['avg_iterations'] <= traditional_summary['avg_iterations']
        assert kalman_summary['convergence_rate'] >= traditional_summary['convergence_rate']
        
    def test_solution_quality_comparison(self):
        """Compare solution quality between methods"""
        solver = EnhancedZ3Solver()
        
        # Challenging scenario
        market_cap = 500_000_000
        profitable_users = 85
        constraints = SolverConstraints(
            min_supply=900_000_000,
            max_supply=1_100_000_000,
            min_price=0.4,
            max_price=0.6,
            min_airdrop_percent=18,
            max_airdrop_percent=25
        )
        
        # Solve with Kalman
        solution_kalman = solver.solve_incremental_with_kalman(
            market_cap, profitable_users, constraints, max_iterations=15
        )
        
        # Solve with traditional
        levels = [
            (100, constraints),
            (50, SolverConstraints(min_airdrop_percent=15, max_airdrop_percent=30)),
            (10, SolverConstraints())
        ]
        solution_traditional = solver.solve_incremental_with_relaxation(
            market_cap, profitable_users, levels
        )
        
        # Calculate qualities
        quality_kalman = 0
        quality_traditional = 0
        
        if solution_kalman:
            quality_kalman = solver._calculate_solution_quality(
                solution_kalman, market_cap, profitable_users
            )
            
        if solution_traditional:
            quality_traditional = solver._calculate_solution_quality(
                solution_traditional, market_cap, profitable_users
            )
        
        perf_logger.info(f"\nQuality Comparison:")
        perf_logger.info(f"  Kalman quality: {quality_kalman:.3f}")
        perf_logger.info(f"  Traditional quality: {quality_traditional:.3f}")
        
        # Kalman should achieve at least as good quality
        assert quality_kalman >= quality_traditional * 0.95  # Allow 5% tolerance
        
    def test_learning_capability(self):
        """Test that Kalman filter improves with experience"""
        solver = EnhancedZ3Solver()
        
        # Similar scenarios to test learning
        base_constraints = SolverConstraints(
            min_supply=800_000_000,
            max_supply=1_200_000_000,
            min_price=0.3,
            max_price=0.7,
            min_airdrop_percent=15,
            max_airdrop_percent=35
        )
        
        # Run multiple similar problems
        qualities = []
        solve_times = []
        
        for i in range(5):
            # Slightly vary the target
            market_cap = 450_000_000 + i * 20_000_000
            profitable_users = 78 + i
            
            start_time = time.time()
            solution = solver.solve_incremental_with_kalman(
                market_cap, profitable_users, base_constraints, max_iterations=10
            )
            solve_time = time.time() - start_time
            
            if solution:
                quality = solver._calculate_solution_quality(solution, market_cap, profitable_users)
                qualities.append(quality)
                solve_times.append(solve_time)
        
        # Performance should improve or stabilize
        if len(qualities) >= 3:
            avg_early = np.mean(qualities[:2])
            avg_late = np.mean(qualities[-2:])
            
            perf_logger.info(f"\nLearning Analysis:")
            perf_logger.info(f"  Early average quality: {avg_early:.3f}")
            perf_logger.info(f"  Late average quality: {avg_late:.3f}")
            
            # Quality should not degrade
            assert avg_late >= avg_early * 0.95
            
    def test_robustness_comparison(self):
        """Test robustness to difficult/infeasible problems"""
        solver = EnhancedZ3Solver()
        
        # Nearly infeasible scenario
        constraints = SolverConstraints(
            min_supply=1_000_000_000,
            max_supply=1_000_000_000,  # Fixed supply
            min_price=0.5,
            max_price=0.5,  # Fixed price (market cap = 500M)
            min_airdrop_percent=40,  # High minimum
            max_airdrop_percent=40   # Fixed airdrop
        )
        
        # These constraints make it very hard to achieve high profitable users
        market_cap = 500_000_000
        profitable_users = 95  # Very high target
        
        # Test both methods
        solution_kalman = solver.solve_incremental_with_kalman(
            market_cap, profitable_users, constraints, max_iterations=15
        )
        
        levels = [(100, constraints), (50, SolverConstraints()), (10, SolverConstraints())]
        solution_traditional = solver.solve_incremental_with_relaxation(
            market_cap, profitable_users, levels
        )
        
        # Both might fail, but Kalman should handle it more gracefully
        perf_logger.info(f"\nRobustness Test:")
        perf_logger.info(f"  Kalman found solution: {solution_kalman is not None}")
        perf_logger.info(f"  Traditional found solution: {solution_traditional is not None}")
        
        # If Kalman found a solution, it should be reasonable
        if solution_kalman:
            assert solution_kalman.total_supply > 0
            assert solution_kalman.launch_price > 0
            
    def test_resource_efficiency(self):
        """Test computational resource usage"""
        solver = EnhancedZ3Solver()
        scenarios = self._generate_test_scenarios()
        
        kalman_total_time = 0
        traditional_total_time = 0
        
        for market_cap, profitable_users, constraints in scenarios[:2]:  # Test first 2 scenarios
            # Time Kalman approach
            start = time.time()
            solver.solve_incremental_with_kalman(
                market_cap, profitable_users, constraints, max_iterations=8
            )
            kalman_total_time += time.time() - start
            
            # Time traditional approach
            start = time.time()
            levels = [
                (100, constraints),
                (50, SolverConstraints(min_airdrop_percent=10, max_airdrop_percent=40)),
                (10, SolverConstraints())
            ]
            solver.solve_incremental_with_relaxation(
                market_cap, profitable_users, levels
            )
            traditional_total_time += time.time() - start
        
        perf_logger.info(f"\nResource Efficiency:")
        perf_logger.info(f"  Kalman total time: {kalman_total_time:.2f}s")
        perf_logger.info(f"  Traditional total time: {traditional_total_time:.2f}s")
        perf_logger.info(f"  Efficiency gain: {traditional_total_time/kalman_total_time:.2f}x")
        
        # Kalman should be competitive or better
        assert kalman_total_time <= traditional_total_time * 1.5  # Allow some overhead


@pytest.mark.benchmark
class TestKalmanBenchmarks:
    """Benchmark tests for Kalman filter components"""
    
    def test_kalman_update_performance(self, benchmark):
        """Benchmark Kalman filter update operation"""
        from airdrop_calculator.solver import KalmanConstraintEstimator
        
        kalman = KalmanConstraintEstimator()
        observation = np.array([0.7, 5.0, 0.85])
        
        def update_operation():
            kalman.update(observation, success=True)
        
        # Benchmark the update operation
        result = benchmark(update_operation)
        
        # Should be very fast (< 1ms)
        assert benchmark.stats['mean'] < 0.001
        
    def test_constraint_relaxation_performance(self, benchmark):
        """Benchmark constraint relaxation"""
        from airdrop_calculator.solver import AdaptiveConstraintScheduler
        
        scheduler = AdaptiveConstraintScheduler(SolverConstraints(
            min_supply=1_000_000_000,
            max_supply=2_000_000_000,
            min_airdrop_percent=10,
            max_airdrop_percent=50
        ))
        
        def relaxation_operation():
            scheduler._relax_constraints(0.5)
        
        result = benchmark(relaxation_operation)
        
        # Should be very fast
        assert benchmark.stats['mean'] < 0.0001


if __name__ == '__main__':
    # Run with performance logging enabled
    pytest.main([__file__, '-v', '-s', '-k', 'test_convergence_speed_comparison'])