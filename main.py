#!/usr/bin/env python3

def main():
    """Main entry point for the enhanced airdrop calculator"""
    cli = AirdropCLI()
    sys.exit(cli.run())

def demo():
    """Run a demonstration of the enhanced features"""
    print("Enhanced Airdrop Calculator Demo")
    print("="*60)
    
    # Create sample parameters
    params = AirdropParameters(
        total_supply=1_000_000_000,
        airdrop_percent=30,
        launch_price=0.50,
        opportunity_cost=10,
        volatility=80,
        gas_cost=50,
        campaign_duration=6,
        airdrop_certainty=70,
        revenue_share=10,
        vesting_months=18,
        immediate_unlock=30
    )
    
    # Run analysis
    calculator = AirdropCalculator(params)
    metrics = calculator.calculate_market_metrics()
    
    print(f"\nDemo Results:")
    print(f"  Min Market Cap: ${metrics.min_market_cap/1e6:.1f}M")
    print(f"  Profitable Users: {metrics.profitable_users_percent:.1f}%")
    print(f"  Optimal Capital: ${metrics.optimal_capital:,.0f}")
    
    # Test enhanced solver
    print(f"\nTesting Enhanced Z3 Solver...")
    solver = EnhancedZ3Solver()
    
    # Try non-linear solving
    solution = solver.solve_with_nonlinear_constraints(
        target_market_cap=200_000_000,
        target_profitable_users=60,
        constraints=SolverConstraints()
    )
    
    if solution:
        print("✓ Non-linear solver found solution!")
    
    # Try Pareto optimization
    print(f"\nFinding Pareto optimal solutions...")
    pareto_solutions = solver.find_pareto_optimal_solutions(
        ['market_cap', 'profitability'],
        SolverConstraints(),
        num_solutions=5
    )
    
    print(f"✓ Found {len(pareto_solutions)} Pareto optimal solutions")
    
    print("\nDemo complete! Use 'python airdrop_calculator.py --help' for CLI usage.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        demo()