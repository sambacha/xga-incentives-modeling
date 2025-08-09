#!/usr/bin/env python3
"""Test track visualization functionality"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from airdrop_calculator.types import TrackType
from airdrop_calculator.track_visualizations import TrackVisualizer
import matplotlib.pyplot as plt

# Create test track results data
test_track_results = [
    {
        'track_type': TrackType.NODE_OPERATOR.name,
        'points': 56.71,
        'capital_equivalent': 56715.00,
        'estimated_allocation': 357223.67,
        'risk_factor': 0.8,
        'roi': -37.56
    },
    {
        'track_type': TrackType.RISK_UNDERWRITER.name,
        'points': 30000.00,
        'capital_equivalent': 10000.00,
        'estimated_allocation': 15000.00,
        'risk_factor': 1.2,
        'roi': -85.71
    },
    {
        'track_type': TrackType.LIQUIDITY_PROVIDER.name,
        'points': 6.00,
        'capital_equivalent': 15000.00,
        'estimated_allocation': 183711.73,
        'risk_factor': 1.0,
        'roi': 18.52
    },
    {
        'track_type': TrackType.AUCTION_PARTICIPANT.name,
        'points': 34000.00,
        'capital_equivalent': 1680.00,
        'estimated_allocation': 6048.00,
        'risk_factor': 1.0,
        'roi': -72.26
    }
]

# Test visualization creation
try:
    print("Creating TrackVisualizer...")
    visualizer = TrackVisualizer()
    
    print("Generating track performance dashboard...")
    visualizer.plot_track_performance_dashboard(
        test_track_results, 
        save_path="test_track_dashboard.png"
    )
    print("✓ Dashboard saved to test_track_dashboard.png")
    
    # Test individual chart methods
    print("\nTesting individual chart methods...")
    
    # Test allocation distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    visualizer._plot_track_allocation_distribution(ax, test_track_results)
    plt.savefig("test_allocation_pie.png")
    plt.close()
    print("✓ Allocation pie chart saved")
    
    # Test points distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    visualizer._plot_track_points_distribution(ax, test_track_results)
    plt.savefig("test_points_bar.png")
    plt.close()
    print("✓ Points bar chart saved")
    
    # Test ROI comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer._plot_track_roi_comparison(ax, test_track_results)
    plt.savefig("test_roi_comparison.png")
    plt.close()
    print("✓ ROI comparison saved")
    
    print("\nAll tests completed successfully!")
    
except Exception as e:
    print(f"\nError during testing: {e}")
    import traceback
    traceback.print_exc()