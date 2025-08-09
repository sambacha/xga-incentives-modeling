#!/usr/bin/env python3
"""Test the fixed chart functionality."""

import tempfile
from pathlib import Path
from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.visualization import AirdropVisualizer

def test_fixed_charts():
    """Test the chart fixes."""
    print("Testing fixed chart generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize components
        solver = EnhancedZ3Solver()
        calculator = AirdropCalculator(solver)
        visualizer = AirdropVisualizer(calculator)
        
        # Test comprehensive analysis
        try:
            plot_file = temp_path / "test_comprehensive_fixed.png"
            visualizer.plot_comprehensive_analysis(save_path=str(plot_file))
            if plot_file.exists():
                print(f"✓ Comprehensive analysis: {plot_file.stat().st_size} bytes")
            else:
                print("✗ Comprehensive analysis failed to create file")
        except Exception as e:
            print(f"✗ Comprehensive analysis error: {e}")
        
        # Test focused analysis
        try:
            plot_file = temp_path / "test_focused_fixed.png"
            visualizer.plot_focused_analysis(save_path=str(plot_file))
            if plot_file.exists():
                print(f"✓ Focused analysis: {plot_file.stat().st_size} bytes")
            else:
                print("✗ Focused analysis failed to create file")
        except Exception as e:
            print(f"✗ Focused analysis error: {e}")
        
        # Test advanced risk analysis
        try:
            plot_file = temp_path / "test_advanced_fixed.png"
            visualizer.plot_advanced_risk_analysis(save_path=str(plot_file))
            if plot_file.exists():
                print(f"✓ Advanced risk analysis: {plot_file.stat().st_size} bytes")
            else:
                print("✗ Advanced risk analysis failed to create file")
        except Exception as e:
            print(f"✗ Advanced risk analysis error: {e}")

if __name__ == "__main__":
    test_fixed_charts()