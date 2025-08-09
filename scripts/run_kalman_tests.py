#!/usr/bin/env python3
"""
Run all Kalman filter tests with a summary report.
"""

import subprocess
import sys
from pathlib import Path

def run_test_suite(test_file, test_name):
    """Run a test suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print('='*60)
    
    cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Run all Kalman filter tests"""
    print("KALMAN FILTER TEST SUITE")
    print("="*80)
    
    test_dir = Path("tests")
    test_suites = [
        ("test_kalman_filter.py", "Unit Tests"),
        ("test_kalman_performance.py", "Performance Tests"),
        ("test_kalman_edge_cases.py", "Edge Case Tests")
    ]
    
    results = {}
    
    for test_file, test_name in test_suites:
        test_path = test_dir / test_file
        if test_path.exists():
            results[test_name] = run_test_suite(str(test_path), test_name)
        else:
            print(f"Warning: {test_path} not found")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for passed in results.values() if passed)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests < total_tests:
        sys.exit(1)
    
    print("\n✓ All Kalman filter tests passed!")
    
    # Run a specific performance comparison if all tests pass
    print("\n" + "="*80)
    print("RUNNING PERFORMANCE COMPARISON")
    print("="*80)
    
    subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/test_kalman_performance.py::TestKalmanPerformanceComparison::test_convergence_speed_comparison",
        "-v", "-s"
    ])

if __name__ == "__main__":
    main()