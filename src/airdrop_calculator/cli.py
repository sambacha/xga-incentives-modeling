import click
import json
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .types import (AirdropParameters, SolverConstraints, ValidationError, SolverError, 
                   TrackType, TrackParameters, NodeOperatorParameters, RiskUnderwriterParameters,
                   LiquidityProviderParameters, AuctionParticipantParameters)
from .core import AirdropCalculator
from .solver import EnhancedZ3Solver
from .visualization import AirdropVisualizer, managed_figure
from .reports import generate_recommendation_report
from .analysis import ProfitabilityAnalyzer
from .tracks import MultiTrackCalculator
from .track_optimizer import TrackOptimizer

logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """
    Enhanced Airdrop Parameter Calculator with Comprehensive Analysis
    
    This tool provides detailed calculations for airdrop economics using
    exotic options theory. All commands now output comprehensive calculation
    results including:
    
    • Step-by-step option pricing formulas (Beta, Hurdle Rate)
    • Detailed investment analysis for multiple scenarios  
    • Market context and token categorization
    • Profitability thresholds and ROI calculations
    • User segment analysis with allocation models
    • Cliff value and break-even calculations
    • Sensitivity analysis for key parameters
    
    Commands:
      calculate  - Show comprehensive calculation results
      analyze    - Analyze parameters with enhanced output
      solve      - Find optimal parameters with detailed results
      pareto     - Find Pareto optimal solutions
      charts     - Generate visualization charts
      analyze-profitability - Analyze investment profitability
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

def _print_comprehensive_calculations(calculator, params, metrics):
    """Print all comprehensive calculation results"""
    click.echo("\n" + "="*80)
    click.echo("COMPREHENSIVE AIRDROP CALCULATION RESULTS")
    click.echo("="*80)
    
    # Basic Parameters
    click.echo(f"\n📊 CORE PARAMETERS:")
    click.echo(f"  Total Supply: {params.total_supply:,.0f} tokens")
    click.echo(f"  Airdrop Percentage: {params.airdrop_percent:.2f}%")
    click.echo(f"  Launch Price: ${params.launch_price:.6f}")
    click.echo(f"  Market Cap: ${params.total_supply * params.launch_price:,.0f}")
    click.echo(f"  Airdrop Tokens: {params.total_supply * (params.airdrop_percent/100):,.0f}")
    click.echo(f"  Airdrop Value: ${params.total_supply * (params.airdrop_percent/100) * params.launch_price:,.0f}")
    
    # Economic Parameters
    click.echo(f"\n💰 ECONOMIC PARAMETERS:")
    click.echo(f"  Opportunity Cost: {params.opportunity_cost:.2f}%")
    click.echo(f"  Volatility: {params.volatility:.2f}%")
    click.echo(f"  Gas Cost: ${params.gas_cost:.2f}")
    click.echo(f"  Campaign Duration: {params.campaign_duration} months")
    click.echo(f"  Airdrop Certainty: {params.airdrop_certainty:.1f}%")
    click.echo(f"  Revenue Share: {params.revenue_share:.1f}%")
    
    # Option Pricing Calculations
    click.echo(f"\n🔬 OPTION PRICING CALCULATIONS:")
    r = params.opportunity_cost / 100
    sigma = params.volatility / 100
    sigma_squared = sigma ** 2
    
    # Show step-by-step beta calculation
    delta = 0.0  # Consistent with our fixes
    a = 0.5 - (r - delta) / sigma_squared
    discriminant = a**2 + 2 * r / sigma_squared
    beta = a + (discriminant**0.5) if discriminant >= 0 else 1.5
    beta = max(1.01, beta)
    
    click.echo(f"  Risk-free rate (r): {r:.4f}")
    click.echo(f"  Volatility (σ): {sigma:.4f}")
    click.echo(f"  Dividend yield (δ): {delta:.4f}")
    click.echo(f"  a parameter: {a:.6f}")
    click.echo(f"  Discriminant: {discriminant:.6f}")
    click.echo(f"  Beta (β): {beta:.6f}")
    
    # Hurdle rate calculation
    epsilon = 1e-9
    hurdle_rate = beta / (beta - 1 + epsilon)
    hurdle_rate = max(1.1, min(hurdle_rate, 10.0))
    
    click.echo(f"  Hurdle Rate: {hurdle_rate:.6f}")
    
    # Market Metrics
    click.echo(f"\n📈 MARKET METRICS:")
    click.echo(f"  Minimum Market Cap: ${metrics.min_market_cap:,.0f}")
    click.echo(f"  Optimal Capital: ${metrics.optimal_capital:,.0f}")
    click.echo(f"  Profitable Users: {metrics.profitable_users_percent:.1f}%")
    click.echo(f"  Average ROI: {metrics.avg_roi:.1f}%")
    click.echo(f"  Typical User Break-even: {metrics.typical_user_break_even:,.0f} tokens")
    
    # Detailed Segment Analysis
    click.echo(f"\n👥 USER SEGMENT ANALYSIS:")
    for segment in metrics.segment_results:
        status = "✅ Profitable" if segment.profitable else "❌ Unprofitable"
        click.echo(f"  {segment.segment}:")
        click.echo(f"    Status: {status}")
        click.echo(f"    Average Capital: ${segment.avg_capital:,.0f}")
        click.echo(f"    Min Tokens Needed: {segment.min_tokens:,.0f}")
        click.echo(f"    Estimated Allocation: {segment.estimated_allocation:,.0f}")
        click.echo(f"    ROI: {segment.roi:.1f}%")
        click.echo(f"    Population: {segment.population_percent:.1f}%")
    
    # Cliff Value Calculations
    click.echo(f"\n⛰️ CLIFF VALUE CALCULATIONS:")
    typical_capital = 5000
    cliff_value = calculator.calculate_cliff_value(
        typical_capital, params.opportunity_cost, params.volatility,
        params.campaign_duration, params.gas_cost
    )
    click.echo(f"  For ${typical_capital:,.0f} capital:")
    click.echo(f"    Opportunity Cost: ${typical_capital * (params.opportunity_cost/100) * (params.campaign_duration/12):,.2f}")
    click.echo(f"    Transaction Costs: ${params.gas_cost * 10 * 1.2:,.2f}")
    click.echo(f"    Risk-Adjusted Cliff: ${cliff_value:,.0f}")
    click.echo(f"    Min Token Price: ${cliff_value / typical_capital:.6f}")
    
    # Allocation Model Results
    click.echo(f"\n🎯 ALLOCATION MODEL RESULTS:")
    allocation_models = ["linear", "quadratic", "logarithmic", "tiered"]
    test_capitals = [1000, 10000, 100000]
    
    for model in allocation_models:
        click.echo(f"  {model.title()} Model:")
        for capital in test_capitals:
            try:
                allocation = calculator.estimate_user_allocation(
                    capital, model, params.total_supply, params.airdrop_percent
                )
                click.echo(f"    ${capital:,.0f} → {allocation:,.0f} tokens")
            except Exception as e:
                click.echo(f"    ${capital:,.0f} → Error: {str(e)}")

def _print_metrics(metrics):
    """Print metrics in a formatted way"""
    click.echo("\n" + "="*60)
    click.echo("AIRDROP ANALYSIS RESULTS")
    click.echo("="*60)
    click.echo(f"\nKey Metrics:")
    click.echo(f"  Minimum Market Cap: ${metrics.min_market_cap/1e6:.1f}M")
    click.echo(f"  Hurdle Rate: {metrics.hurdle_rate:.2f}x")
    click.echo(f"  Beta Value: {metrics.beta_value:.3f}")
    click.echo(f"  Profitable Users: {metrics.profitable_users_percent:.1f}%")
    click.echo(f"  Average ROI: {metrics.avg_roi:.1f}%")
    click.echo(f"  Optimal Capital: ${metrics.optimal_capital:,.0f}")
    
    click.echo(f"\nSegment Analysis:")
    for segment in metrics.segment_results:
        status = "✓ Profitable" if segment.profitable else "✗ Unprofitable"
        click.echo(f"  {segment.segment}: {status} (ROI: {segment.roi:.1f}%)")

def _print_solution(solution: AirdropParameters):
    """Print solution parameters with detailed calculations"""
    click.echo("\n" + "="*80)
    click.echo("OPTIMAL SOLUTION WITH DETAILED CALCULATIONS")
    click.echo("="*80)
    
    # Core Solution Parameters
    click.echo(f"\n📊 CORE SOLUTION PARAMETERS:")
    click.echo(f"  Total Supply: {solution.total_supply:,.0f} tokens")
    click.echo(f"  Airdrop Percentage: {solution.airdrop_percent:.2f}%")
    click.echo(f"  Launch Price: ${solution.launch_price:.6f}")
    click.echo(f"  Market Cap: ${solution.total_supply * solution.launch_price:,.0f}")
    click.echo(f"  Airdrop Tokens: {solution.total_supply * (solution.airdrop_percent/100):,.0f}")
    click.echo(f"  Airdrop Value: ${solution.total_supply * (solution.airdrop_percent/100) * solution.launch_price:,.0f}")
    
    # Economic Parameters
    click.echo(f"\n💰 ECONOMIC PARAMETERS:")
    click.echo(f"  Opportunity Cost: {solution.opportunity_cost:.2f}%")
    click.echo(f"  Volatility: {solution.volatility:.2f}%")
    click.echo(f"  Gas Cost: ${solution.gas_cost:.2f}")
    click.echo(f"  Campaign Duration: {solution.campaign_duration} months")
    click.echo(f"  Airdrop Certainty: {solution.airdrop_certainty:.1f}%")
    click.echo(f"  Revenue Share: {solution.revenue_share:.1f}%")
    click.echo(f"  Vesting: {solution.vesting_months} months")
    click.echo(f"  Immediate Unlock: {solution.immediate_unlock:.1f}%")
    
    # Option Pricing Calculations
    click.echo(f"\n🔬 OPTION PRICING CALCULATIONS:")
    if solution.beta and solution.hurdle_rate:
        click.echo(f"  Beta (β): {solution.beta:.6f}")
        click.echo(f"  Hurdle Rate: {solution.hurdle_rate:.6f}")
        
        # Show the calculation breakdown
        r = solution.opportunity_cost / 100
        sigma = solution.volatility / 100
        click.echo(f"  Risk-free rate (r): {r:.4f}")
        click.echo(f"  Volatility (σ): {sigma:.4f}")
        click.echo(f"  Formula: β = 0.5 - (r-δ)/σ² + √[(0.5 - (r-δ)/σ²)² + 2r/σ²]")
        click.echo(f"  Hurdle Formula: H = β/(β-1) = {solution.hurdle_rate:.6f}")
    else:
        click.echo(f"  Beta and Hurdle Rate: Not calculated in this solution")
    
    # Investment Analysis
    click.echo(f"\n💡 INVESTMENT ANALYSIS:")
    typical_investments = [1000, 5000, 10000, 50000]
    
    for investment in typical_investments:
        tokens_per_dollar = 1 / solution.launch_price
        tokens_received = investment * tokens_per_dollar
        
        # Estimate allocation based on quadratic model
        total_airdrop = solution.total_supply * (solution.airdrop_percent / 100)
        # Simplified allocation estimation
        allocation_ratio = (investment / 500_000) * 0.01  # Very rough estimate
        estimated_allocation = total_airdrop * allocation_ratio
        
        click.echo(f"  ${investment:,.0f} investment:")
        click.echo(f"    Tokens if purchased: {tokens_received:,.0f}")
        click.echo(f"    Est. airdrop allocation: {estimated_allocation:,.0f}")
        click.echo(f"    Allocation ratio: {(estimated_allocation/tokens_received)*100:.2f}%" if tokens_received > 0 else "    Allocation ratio: N/A")
    
    # Profitability Thresholds
    click.echo(f"\n⚖️ PROFITABILITY THRESHOLDS:")
    if solution.hurdle_rate:
        click.echo(f"  Required return multiple: {solution.hurdle_rate:.2f}x")
        min_profitable_price = solution.launch_price * solution.hurdle_rate
        click.echo(f"  Min profitable exit price: ${min_profitable_price:.6f}")
        
        # Calculate for different holding periods
        for months in [3, 6, 12, 24]:
            annual_return = (solution.hurdle_rate - 1) * (12 / months)
            click.echo(f"  {months}m holding period: {annual_return*100:.1f}% annual return needed")
    
    # Token Distribution Breakdown
    click.echo(f"\n🎯 TOKEN DISTRIBUTION BREAKDOWN:")
    airdrop_tokens = solution.total_supply * (solution.airdrop_percent / 100)
    immediate_tokens = airdrop_tokens * (solution.immediate_unlock / 100)
    vested_tokens = airdrop_tokens - immediate_tokens
    
    click.echo(f"  Total Airdrop: {airdrop_tokens:,.0f} tokens ({solution.airdrop_percent:.1f}%)")
    click.echo(f"  Immediate Release: {immediate_tokens:,.0f} tokens ({solution.immediate_unlock:.1f}%)")
    click.echo(f"  Vested Amount: {vested_tokens:,.0f} tokens (over {solution.vesting_months} months)")
    click.echo(f"  Monthly Vesting: {vested_tokens / solution.vesting_months:,.0f} tokens")
    
    # Market Comparison
    click.echo(f"\n📊 MARKET CONTEXT:")
    market_cap = solution.total_supply * solution.launch_price
    
    if market_cap < 10_000_000:
        size_category = "Micro-cap"
    elif market_cap < 100_000_000:
        size_category = "Small-cap"
    elif market_cap < 1_000_000_000:
        size_category = "Mid-cap"
    else:
        size_category = "Large-cap"
    
    click.echo(f"  Market Cap Category: {size_category}")
    click.echo(f"  Price per token: ${solution.launch_price:.6f}")
    click.echo(f"  Tokens per $1: {1/solution.launch_price:,.2f}")
    click.echo(f"  FDV (Fully Diluted): ${market_cap:,.0f}")
    
    if solution.penalties:
        click.echo("\n" + "="*60)
        click.echo("⚠️ CONSTRAINT PENALTY REPORT")
        click.echo("="*60)
        click.echo("The following trade-offs were made to find the best-effort solution:")
        for name, penalty in solution.penalties.items():
            if penalty > 0.01:
                click.echo(f"  - {name}: Penalty of {penalty:.3f}")
                if name == 'market_cap':
                    click.echo(f"    (Market cap target was not fully achieved)")
                elif name == 'profitable_users':
                    click.echo(f"    (Profitable users target was not fully achieved)")
    
    click.echo("\n" + "="*80)

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Load from JSON config')
@click.option('--supply', type=float, help='Total token supply')
@click.option('--airdrop-percent', type=float, help='Airdrop percentage')
@click.option('--price', type=float, help='Launch price')
@click.option('--opportunity-cost', type=float, default=10)
@click.option('--volatility', type=float, default=80)
@click.option('--gas-cost', type=float, default=50)
@click.option('--duration', type=int, default=6)
@click.option('--certainty', type=float, default=70)
@click.option('--revenue-share', type=float, default=10)
@click.option('--output', type=click.Path(), help='Output directory for detailed results')
def calculate(config, supply, airdrop_percent, price, opportunity_cost, volatility, gas_cost, duration, certainty, revenue_share, output):
    """Show comprehensive calculation results for airdrop parameters"""
    try:
        if config:
            with open(config, 'r') as f:
                params_dict = json.load(f)
            params = AirdropParameters(**params_dict)
        else:
            if not all([supply, airdrop_percent, price]):
                raise click.UsageError("Must provide --supply, --airdrop-percent, and --price or use --config")
            
            params = AirdropParameters(
                total_supply=supply, airdrop_percent=airdrop_percent, launch_price=price,
                opportunity_cost=opportunity_cost, volatility=volatility, gas_cost=gas_cost,
                campaign_duration=duration, airdrop_certainty=certainty,
                revenue_share=revenue_share, vesting_months=18, immediate_unlock=30
            )
        
        calculator = AirdropCalculator(params)
        metrics = calculator.calculate_market_metrics()
        
        # Print comprehensive calculations
        _print_comprehensive_calculations(calculator, params, metrics)
        
        # Additional detailed calculations
        click.echo(f"\n🔍 ADDITIONAL CALCULATIONS:")
        
        # Test different allocation scenarios
        click.echo(f"\n📊 ALLOCATION SCENARIO TESTING:")
        test_scenarios = [
            ("Conservative", 1000, 5000),
            ("Moderate", 5000, 25000),
            ("Aggressive", 25000, 100000)
        ]
        
        for scenario_name, min_cap, max_cap in test_scenarios:
            avg_cap = (min_cap + max_cap) / 2
            min_tokens = calculator.calculate_min_profitable_tokens(avg_cap)
            estimated_allocation = calculator.estimate_user_allocation(
                avg_cap, "quadratic", params.total_supply, params.airdrop_percent
            )
            profitable = estimated_allocation >= min_tokens
            
            click.echo(f"  {scenario_name} User (${avg_cap:,.0f}):")
            click.echo(f"    Min tokens needed: {min_tokens:,.0f}")
            click.echo(f"    Estimated allocation: {estimated_allocation:,.0f}")
            click.echo(f"    Profitable: {'✅ Yes' if profitable else '❌ No'}")
            
            if profitable:
                roi = ((estimated_allocation * params.launch_price) - avg_cap) / avg_cap * 100
                click.echo(f"    Estimated ROI: {roi:.1f}%")
        
        # Option pricing sensitivity analysis
        click.echo(f"\n📈 OPTION PRICING SENSITIVITY:")
        base_opp_cost = params.opportunity_cost
        base_volatility = params.volatility
        
        for opp_cost in [base_opp_cost * 0.5, base_opp_cost, base_opp_cost * 1.5]:
            for vol in [base_volatility * 0.8, base_volatility, base_volatility * 1.2]:
                beta = calculator.calculate_beta(opp_cost/100, vol/100)
                hurdle = calculator.calculate_hurdle_rate(opp_cost, vol)
                click.echo(f"  OppCost {opp_cost:.1f}%, Vol {vol:.1f}%: β={beta:.3f}, H={hurdle:.3f}")
        
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            detailed_results = {
                'parameters': params.__dict__,
                'metrics': {k: v for k, v in metrics.__dict__.items() if k != 'segment_results'},
                'segment_analysis': [seg.__dict__ for seg in metrics.segment_results],
                'calculations': {
                    'market_cap': params.total_supply * params.launch_price,
                    'airdrop_value': params.total_supply * (params.airdrop_percent/100) * params.launch_price,
                    'tokens_per_dollar': 1 / params.launch_price,
                    'option_pricing': {
                        'risk_free_rate': params.opportunity_cost / 100,
                        'volatility': params.volatility / 100,
                        'beta': metrics.beta_value,
                        'hurdle_rate': metrics.hurdle_rate
                    }
                }
            }
            
            results_file = output_dir / 'comprehensive_calculations.json'
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            click.echo(f"\n💾 Detailed results saved to: {results_file}")
        
    except (ValidationError, SolverError, click.UsageError) as e:
        logger.error(f"Error: {e}")
        raise click.Abort()

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Load from JSON config')
@click.option('--supply', type=float, help='Total token supply')
@click.option('--airdrop-percent', type=float, help='Airdrop percentage')
@click.option('--price', type=float, help='Launch price')
@click.option('--opportunity-cost', type=float, default=10)
@click.option('--volatility', type=float, default=80)
@click.option('--gas-cost', type=float, default=50)
@click.option('--duration', type=int, default=6)
@click.option('--certainty', type=float, default=70)
@click.option('--plot', is_flag=True, help='Generate basic plots')
@click.option('--charts', type=click.Choice(['basic', 'advanced', 'focused', 'all'], case_sensitive=False), 
              default='basic', help='Chart generation mode')
@click.option('--chart-format', type=click.Choice(['png', 'pdf', 'svg'], case_sensitive=False), 
              default='png', help='Chart output format')
@click.option('--report', is_flag=True, help='Generate a recommendation report')
@click.option('--output', type=click.Path(), help='Output directory')
def analyze(config, supply, airdrop_percent, price, opportunity_cost, volatility, gas_cost, duration, certainty, plot, charts, chart_format, report, output):
    """Analyze airdrop parameters"""
    try:
        if config:
            with open(config, 'r') as f:
                params_dict = json.load(f)
            params = AirdropParameters(**params_dict)
        else:
            if not all([supply, airdrop_percent, price]):
                raise click.UsageError("Must provide --supply, --airdrop-percent, and --price or use --config")
            
            params = AirdropParameters(
                total_supply=supply, airdrop_percent=airdrop_percent, launch_price=price,
                opportunity_cost=opportunity_cost, volatility=volatility, gas_cost=gas_cost,
                campaign_duration=duration, airdrop_certainty=certainty,
                revenue_share=10, vesting_months=18, immediate_unlock=30
            )
        
        calculator = AirdropCalculator(params)
        metrics = calculator.calculate_market_metrics()
        
        # Print comprehensive calculations
        _print_comprehensive_calculations(calculator, params, metrics)
        
        output_dir = Path(output) if output else Path('.')
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / 'analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'parameters': params.__dict__,
                'metrics': {k: v for k, v in metrics.__dict__.items() if k != 'segment_results'}
            }, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        if plot or charts != 'basic':
            visualizer = AirdropVisualizer(calculator)
            
            # Generate charts based on the selected mode
            if charts == 'all':
                chart_results = visualizer.generate_all_charts(str(output_dir))
                click.echo(f"Generated all chart types:")
                for chart_type, path in chart_results.items():
                    if path:
                        click.echo(f"  ✓ {chart_type}: {path}")
                    else:
                        click.echo(f"  ✗ {chart_type}: Failed to generate")
                        
            elif charts == 'advanced':
                plot_file = output_dir / f'advanced_risk_analysis.{chart_format}'
                visualizer.plot_advanced_risk_analysis(save_path=str(plot_file))
                click.echo(f"Advanced risk analysis saved to {plot_file}")
                
            elif charts == 'focused':
                plot_file = output_dir / f'focused_analysis.{chart_format}'
                visualizer.plot_focused_analysis(save_path=str(plot_file))
                click.echo(f"Focused analysis saved to {plot_file}")
                
            else:  # basic or plot flag
                plot_file = output_dir / f'comprehensive_analysis.{chart_format}'
                visualizer.plot_comprehensive_analysis(save_path=str(plot_file))
                click.echo(f"Comprehensive analysis saved to {plot_file}")
        
        if report:
            report_file = output_dir / 'recommendations.md'
            generate_recommendation_report(params, calculator, str(report_file))
            logger.info(f"Report saved to {report_file}")

    except (ValidationError, SolverError, click.UsageError) as e:
        logger.error(f"Error: {e}")
        raise click.Abort()

@cli.command()
@click.option('--market-cap', type=float, required=True)
@click.option('--profitable-users', type=float, required=True)
@click.option('--method', type=click.Choice(['nonlinear', 'soft', 'incremental', 'kalman']), default='nonlinear')
@click.option('--constraints', type=click.Path(exists=True), help='Constraints JSON file')
@click.option('--output', type=click.Path(), help='Output file for the solution')
def solve(market_cap, profitable_users, method, constraints, output):
    """Solve for optimal parameters"""
    solver = EnhancedZ3Solver()
    
    if constraints:
        with open(constraints, 'r') as f:
            constraints_dict = json.load(f)
        constraints_obj = SolverConstraints(**constraints_dict)
    else:
        constraints_obj = SolverConstraints()
    
    logger.info(f"Solving with {method} method...")
    
    solution = None
    if method == 'nonlinear':
        solution = solver.solve_with_nonlinear_constraints(market_cap, profitable_users, constraints_obj)
    elif method == 'soft':
        objectives = {'market_cap': (market_cap, 1000), 'profitable_users': (profitable_users, 100)}
        solution = solver.solve_with_soft_constraints(objectives, constraints_obj)
    elif method == 'incremental':
        levels = [
            (100, constraints_obj),
            (50, SolverConstraints(min_airdrop_percent=10, max_airdrop_percent=40)),
            (10, SolverConstraints())
        ]
        solution = solver.solve_incremental_with_relaxation(market_cap, profitable_users, levels)
    elif method == 'kalman':
        solution = solver.solve_incremental_with_kalman(market_cap, profitable_users, constraints_obj)
    
    if solution:
        logger.info("Solution found!")
        _print_solution(solution)
        if output:
            # Create a copy of the solution to avoid modifying the original
            solution_dict = solution.__dict__.copy()
            # The penalties are for display only, so we don't save them
            solution_dict.pop('penalties', None)
            with open(output, 'w') as f:
                json.dump(solution_dict, f, indent=2)
            logger.info(f"Solution saved to {output}")
    else:
        logger.error("No solution found!")
        raise click.Abort()

@cli.command()
@click.option('--objectives', required=True, help='Comma-separated objectives (e.g., market_cap,profitability)')
@click.option('--constraints', type=click.Path(exists=True), help='Constraints JSON file')
@click.option('--num-solutions', default=20, type=int)
@click.option('--output', type=click.Path(), help='Output file for solutions')
def pareto(objectives, constraints, num_solutions, output):
    """Find Pareto optimal solutions"""
    solver = EnhancedZ3Solver()
    objectives_list = objectives.split(',')
    
    if constraints:
        with open(constraints, 'r') as f:
            constraints_dict = json.load(f)
        constraints_obj = SolverConstraints(**constraints_dict)
    else:
        constraints_obj = SolverConstraints()
        
    logger.info(f"Finding Pareto optimal solutions for objectives: {objectives_list}")
    
    solutions = solver.find_pareto_optimal_solutions(objectives_list, constraints_obj, num_solutions)
    
    if solutions:
        logger.info(f"Found {len(solutions)} Pareto optimal solutions")
        click.echo(json.dumps(solutions, indent=2))
        if output:
            with open(output, 'w') as f:
                json.dump(solutions, f, indent=2)
            logger.info(f"Solutions saved to {output}")
    else:
        logger.error("No Pareto optimal solutions found!")
        raise click.Abort()

@cli.command()
@click.option('--capital', type=float, required=True, help='Capital invested.')
@click.option('--opportunity-cost', type=float, required=True, help='Opportunity cost rate (e.g., 0.1 for 10%).')
@click.option('--time-months', type=int, default=6, help='Time horizon in months.')
@click.option('--expected-share', type=float, required=True, help='Expected share of the airdrop (e.g., 0.001 for 0.1%).')
@click.option('--total-airdrop', type=float, required=True, help='Total number of tokens in the airdrop.')
def analyze_profitability(capital, opportunity_cost, time_months, expected_share, total_airdrop):
    """Analyze the profitability of an airdrop participation strategy."""
    # --- Scratchpad ---
    # Initial thought: Just create the calculator and analyzer and run the analysis.
    # `solver = EnhancedZ3Solver()`
    # `calculator = AirdropCalculator(solver)`
    # `analyzer = ProfitabilityAnalyzer(calculator)`
    # `result = analyzer.analyze_strategy(...)`
    #
    # But wait, maybe it's better... to instantiate the AirdropCalculator with some
    # default AirdropParameters. The analyzer relies on the calculator, which in turn
    # expects to be initialized with parameters. Although the new analysis functions
    # don't use all of them, it's better to provide a default, valid set of parameters
    # to ensure the calculator is in a consistent state.
    # This also makes the tool more robust if future changes to the analyzer
    # require more parameters from the calculator.

    try:
        solver = EnhancedZ3Solver()
        # We need to initialize the calculator with some default parameters,
        # even if they are not all used by the profitability analyzer.
        default_params = AirdropParameters(
            total_supply=1_000_000_000,
            airdrop_percent=10,
            launch_price=1.0,
            opportunity_cost=10,
            volatility=80,
            gas_cost=500,
            campaign_duration=6,
            airdrop_certainty=70,
            revenue_share=10,
            vesting_months=12,
            immediate_unlock=25
        )
        calculator = AirdropCalculator(solver)
        analyzer = ProfitabilityAnalyzer(calculator)

        result = analyzer.analyze_strategy(
            strategy_name="Custom Analysis",
            capital=capital,
            opportunity_cost_rate=opportunity_cost,
            expected_allocation_share=expected_share,
            total_airdrop_tokens=total_airdrop,
            time_months=time_months,
        )

        click.echo("\n" + "="*60)
        click.echo("PROFITABILITY ANALYSIS")
        click.echo("="*60)
        click.echo(f"Strategy: {result['strategy_name']}")
        click.echo(f"  Capital At Risk: ${result['capital']:,.2f}")
        click.echo(f"  Opportunity Cost Rate: {result['opportunity_cost_rate']:.2%}")
        click.echo(f"  Expected Allocation Share: {result['expected_allocation_share']:.4%}")
        click.echo("-" * 60)
        click.echo(f"Required Token Price: ${result['required_token_price']:.4f}")
        click.echo(f"Required Market Cap: ${result['required_market_cap']:,.2f}")
        click.echo("-" * 60)
        click.echo("Details:")
        click.echo(f"  Hurdle Multiple: {result['details']['hurdle_multiple']:.2f}x")
        click.echo(f"  Total Opportunity Cost: ${result['details']['opportunity_cost']:,.2f}")
        click.echo(f"  Risk-Adjusted Required Value: ${result['details']['risk_adjusted_required_value']:,.2f}")

    except (ValidationError, SolverError, click.UsageError) as e:
        logger.error(f"Error: {e}")
        raise click.Abort()

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Path to JSON config file')
@click.option('--supply', type=float, help='Total token supply')
@click.option('--airdrop-percent', type=float, help='Percentage for airdrop')
@click.option('--price', type=float, help='Launch price per token')
@click.option('--opportunity-cost', type=float, default=10, help='Opportunity cost (%)')
@click.option('--volatility', type=float, default=80, help='Expected volatility (%)')
@click.option('--gas-cost', type=float, default=50, help='Gas cost per transaction')
@click.option('--duration', type=int, default=6, help='Campaign duration (months)')
@click.option('--certainty', type=float, default=70, help='Airdrop certainty (%)')
@click.option('--mode', type=click.Choice(['basic', 'advanced', 'focused', 'comparison', 'all'], case_sensitive=False), 
              default='all', help='Chart generation mode')
@click.option('--format', 'chart_format', type=click.Choice(['png', 'pdf', 'svg'], case_sensitive=False), 
              default='png', help='Output format')
@click.option('--scenarios', type=click.Path(exists=True), help='JSON file with scenarios for comparison')
@click.option('--output', type=click.Path(), default='./charts', help='Output directory')
@click.option('--dpi', type=int, default=300, help='Chart resolution (DPI)')
def charts(config, supply, airdrop_percent, price, opportunity_cost, volatility, gas_cost, 
          duration, certainty, mode, chart_format, scenarios, output, dpi):
    """
    Generate comprehensive airdrop analysis charts
    
    But wait, maybe it's better... to provide detailed help text and examples
    for each chart mode to help users understand what they're getting
    """
    try:
        # Setup parameters
        if config:
            with open(config, 'r') as f:
                params_dict = json.load(f)
            params = AirdropParameters(**params_dict)
        else:
            if not all([supply, airdrop_percent, price]):
                raise click.UsageError("Must provide --supply, --airdrop-percent, and --price or use --config")
            
            params = AirdropParameters(
                total_supply=supply, airdrop_percent=airdrop_percent, launch_price=price,
                opportunity_cost=opportunity_cost, volatility=volatility, gas_cost=gas_cost,
                campaign_duration=duration, airdrop_certainty=certainty,
                revenue_share=10, vesting_months=18, immediate_unlock=30
            )
        
        # Initialize calculator and visualizer
        calculator = AirdropCalculator(params)
        visualizer = AirdropVisualizer(calculator)
        
        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Generating {mode} charts in {chart_format} format...")
        click.echo(f"Output directory: {output_dir}")
        click.echo("-" * 60)
        
        # Load scenarios if provided
        scenario_data = None
        if scenarios:
            with open(scenarios, 'r') as f:
                scenario_data = json.load(f)
        
        # Generate charts based on mode
        if mode == 'basic':
            plot_file = output_dir / f'comprehensive_analysis.{chart_format}'
            visualizer.plot_comprehensive_analysis(save_path=str(plot_file), dpi=dpi)
            click.echo(f"✓ Basic comprehensive analysis: {plot_file}")
            
        elif mode == 'advanced':
            plot_file = output_dir / f'advanced_risk_analysis.{chart_format}'
            visualizer.plot_advanced_risk_analysis(save_path=str(plot_file), dpi=dpi)
            click.echo(f"✓ Advanced risk analysis: {plot_file}")
            
        elif mode == 'focused':
            plot_file = output_dir / f'focused_analysis.{chart_format}'
            visualizer.plot_focused_analysis(save_path=str(plot_file), dpi=dpi)
            click.echo(f"✓ Focused decision analysis: {plot_file}")
            
        elif mode == 'comparison' and scenario_data:
            plot_file = output_dir / f'scenario_comparison.{chart_format}'
            visualizer.plot_scenario_comparison(scenario_data, save_path=str(plot_file), dpi=dpi)
            click.echo(f"✓ Scenario comparison: {plot_file}")
            
        elif mode == 'all':
            chart_results = visualizer.generate_all_charts(str(output_dir), scenario_data)
            click.echo("Generated chart suite:")
            for chart_type, path in chart_results.items():
                if path:
                    click.echo(f"  ✓ {chart_type.replace('_', ' ').title()}: {path}")
                else:
                    click.echo(f"  ✗ {chart_type.replace('_', ' ').title()}: Failed")
        
        else:
            raise click.UsageError(f"Invalid mode '{mode}' or missing scenarios file for comparison mode")
        
        click.echo("-" * 60)
        click.echo(f"Chart generation completed! Check {output_dir} for results.")
        
        # Display chart descriptions
        if mode in ['all', 'advanced']:
            click.echo("\nAdvanced charts include:")
            click.echo("  • Volatility sensitivity heatmap")
            click.echo("  • Beta vs hurdle rate relationships")
            click.echo("  • Capital efficiency contours")
            click.echo("  • Time value decay analysis")
            click.echo("  • User profitability distributions")
            click.echo("  • Risk-reward scatter plots")
            click.echo("  • 3D optimal allocation surfaces")
        
    except (ValidationError, SolverError, click.UsageError) as e:
        logger.error(f"Error: {e}")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise click.Abort()

@cli.command()
@click.option('--track', type=click.Choice(['node-operator', 'risk-underwriter', 'liquidity-provider', 'auction-participant']), 
              required=True, help='Track type to calculate for')
@click.option('--config', type=click.Path(exists=True), help='Path to airdrop parameters config file')
# Node operator options
@click.option('--validators', type=int, help='Number of validators operated (1-100)')
@click.option('--operation-months', type=int, help='Duration of operation in months (1-36)')
@click.option('--performance', type=float, default=1.0, help='Validator performance score (0.8-1.2)')
@click.option('--uptime', type=float, default=99.0, help='Uptime percentage (90-100)')
# Risk underwriter options
@click.option('--tokens-staked', type=float, help='Amount of tokens staked')
@click.option('--staking-months', type=int, help='Staking duration in months (1-48)')
@click.option('--token-type', type=click.Choice(['FOLD', 'EIGEN']), default='FOLD', help='Token type')
# Liquidity provider options
@click.option('--lst-amount', type=float, help='Amount of LSTs provided')
@click.option('--liquidity-months', type=int, help='Liquidity duration in months (1-36)')
@click.option('--pool-type', type=str, default='default', help='Pool type identifier')
@click.option('--pool-bonus', type=float, default=1.0, help='Pool bonus multiplier (1.0-1.5)')
# Auction participant options
@click.option('--total-bids', type=int, help='Total number of bids')
@click.option('--successful-bids', type=int, help='Number of successful bids')
@click.option('--bid-accuracy', type=float, help='Bid accuracy percentage (0-100)')
@click.option('--bid-value', type=float, help='Total bid value')
# Output options
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def track_calculate(track, config, validators, operation_months, performance, uptime,
                   tokens_staked, staking_months, token_type, lst_amount, liquidity_months,
                   pool_type, pool_bonus, total_bids, successful_bids, bid_accuracy, bid_value, json_output):
    """Calculate allocation for a specific track"""
    try:
        # Load airdrop parameters
        if config:
            with open(config, 'r') as f:
                params_dict = json.load(f)
            airdrop_params = AirdropParameters(**params_dict)
        else:
            # Use optimized default parameters
            from .defaults import OPTIMIZED_DEFAULTS
            airdrop_params = AirdropParameters(**OPTIMIZED_DEFAULTS)
        
        # Create track parameters based on track type
        track_params = None
        if track == 'node-operator':
            if not all([validators, operation_months]):
                raise click.UsageError("Node operator track requires --validators and --operation-months")
            track_params = TrackParameters(
                track_type=TrackType.NODE_OPERATOR,
                node_operator=NodeOperatorParameters(
                    validators_operated=validators,
                    operation_duration_months=operation_months,
                    validator_performance_score=performance,
                    uptime_percentage=uptime
                )
            )
        elif track == 'risk-underwriter':
            if not all([tokens_staked, staking_months]):
                raise click.UsageError("Risk underwriter track requires --tokens-staked and --staking-months")
            track_params = TrackParameters(
                track_type=TrackType.RISK_UNDERWRITER,
                risk_underwriter=RiskUnderwriterParameters(
                    tokens_staked=tokens_staked,
                    staking_duration_months=staking_months,
                    token_type=token_type
                )
            )
        elif track == 'liquidity-provider':
            if not all([lst_amount, liquidity_months]):
                raise click.UsageError("Liquidity provider track requires --lst-amount and --liquidity-months")
            track_params = TrackParameters(
                track_type=TrackType.LIQUIDITY_PROVIDER,
                liquidity_provider=LiquidityProviderParameters(
                    lst_amount=lst_amount,
                    liquidity_duration_months=liquidity_months,
                    pool_type=pool_type,
                    pool_bonus_multiplier=pool_bonus
                )
            )
        elif track == 'auction-participant':
            if not all([total_bids, successful_bids, bid_accuracy, bid_value]):
                raise click.UsageError("Auction participant track requires --total-bids, --successful-bids, --bid-accuracy, and --bid-value")
            track_params = TrackParameters(
                track_type=TrackType.AUCTION_PARTICIPANT,
                auction_participant=AuctionParticipantParameters(
                    total_bids=total_bids,
                    successful_bids=successful_bids,
                    bid_accuracy=bid_accuracy,
                    total_bid_value=bid_value
                )
            )
        
        # Calculate allocation
        calculator = AirdropCalculator(airdrop_params)
        result = calculator.calculate_track_allocation(track_params, airdrop_params)
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n{track.replace('-', ' ').title()} Track Analysis")
            click.echo("=" * 60)
            click.echo(f"Points Earned: {result['points']:,.2f}")
            click.echo(f"Capital Equivalent: ${result['capital_equivalent']:,.2f}")
            click.echo(f"Risk Factor: {result['risk_factor']:.2f}x")
            click.echo(f"Allocation Model: {result['allocation_model']}")
            click.echo(f"\nEstimated Allocation: {result['estimated_allocation']:,.2f} tokens")
            click.echo(f"Minimum Profitable: {result['min_profitable_tokens']:,.2f} tokens")
            click.echo(f"Profitable: {'Yes' if result['is_profitable'] else 'No'}")
            click.echo(f"\nFinancial Analysis:")
            click.echo(f"  Gross Value: ${result['gross_value']:,.2f}")
            click.echo(f"  Total Cost: ${result['total_cost']:,.2f}")
            click.echo(f"  ROI: {result['roi']:.2f}%")
            
    except (ValidationError, click.UsageError) as e:
        logger.error(f"Error: {e}")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise click.Abort()

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Path to multi-track configuration file', required=True)
@click.option('--airdrop-config', type=click.Path(exists=True), help='Path to airdrop parameters config file')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def multi_track(config, airdrop_config, json_output):
    """Calculate allocation across multiple tracks"""
    try:
        # Load track configurations
        with open(config, 'r') as f:
            track_configs = json.load(f)
        
        # Load airdrop parameters
        if airdrop_config:
            with open(airdrop_config, 'r') as f:
                params_dict = json.load(f)
            airdrop_params = AirdropParameters(**params_dict)
        else:
            # Use optimized default parameters
            from .defaults import OPTIMIZED_DEFAULTS
            airdrop_params = AirdropParameters(**OPTIMIZED_DEFAULTS)
        
        # Create track parameters
        track_params_list = []
        for track_config in track_configs:
            track_type = track_config['track_type']
            if track_type == 'NODE_OPERATOR':
                params = track_config['parameters']
                track_params = TrackParameters(
                    track_type=TrackType.NODE_OPERATOR,
                    node_operator=NodeOperatorParameters(**params)
                )
            elif track_type == 'RISK_UNDERWRITER':
                params = track_config['parameters']
                track_params = TrackParameters(
                    track_type=TrackType.RISK_UNDERWRITER,
                    risk_underwriter=RiskUnderwriterParameters(**params)
                )
            elif track_type == 'LIQUIDITY_PROVIDER':
                params = track_config['parameters']
                track_params = TrackParameters(
                    track_type=TrackType.LIQUIDITY_PROVIDER,
                    liquidity_provider=LiquidityProviderParameters(**params)
                )
            elif track_type == 'AUCTION_PARTICIPANT':
                params = track_config['parameters']
                track_params = TrackParameters(
                    track_type=TrackType.AUCTION_PARTICIPANT,
                    auction_participant=AuctionParticipantParameters(**params)
                )
            track_params_list.append(track_params)
        
        # Calculate multi-track allocation
        calculator = AirdropCalculator(airdrop_params)
        result = calculator.calculate_multi_track_allocation(track_params_list, airdrop_params)
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("\nMulti-Track Allocation Analysis")
            click.echo("=" * 60)
            click.echo("\nTrack Summary:")
            for track_name, track_data in result['track_summary'].items():
                click.echo(f"\n{track_name.replace('_', ' ').title()}:")
                click.echo(f"  Points: {track_data['points']:,.2f}")
                click.echo(f"  Capital Equivalent: ${track_data['capital_equivalent']:,.2f}")
                click.echo(f"  Risk Factor: {track_data['risk_factor']:.2f}x")
                click.echo(f"  Allocation Model: {track_data['model']}")
            
            click.echo(f"\nTotal Points: {result['total_points']:,.2f}")
            click.echo(f"Total Capital Equivalent: ${result['total_capital_equivalent']:,.2f}")
            click.echo(f"Weighted Risk Factor: {result['weighted_risk_factor']:.2f}x")
            click.echo(f"\nRecommended Total Allocation: {result['recommended_allocation']:,.2f} tokens")
            click.echo(f"Minimum Profitable: {result['min_profitable_tokens']:,.2f} tokens")
            click.echo(f"Profitable: {'Yes' if result['is_profitable'] else 'No'}")
            
            click.echo(f"\nAllocation Breakdown by Track:")
            for track_type, allocation in result['allocation_breakdown'].items():
                click.echo(f"  {track_type.name.replace('_', ' ').title()}: {allocation:,.2f} tokens")
            
            click.echo(f"\nFinancial Analysis:")
            click.echo(f"  Gross Value: ${result['gross_value']:,.2f}")
            click.echo(f"  Total Cost: ${result['total_cost']:,.2f}")
            click.echo(f"  Overall ROI: {result['overall_roi']:.2f}%")
            
    except (ValidationError, click.UsageError) as e:
        logger.error(f"Error: {e}")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise click.Abort()

@cli.command()
@click.option('--scenarios', type=click.Path(exists=True), help='Path to track scenarios configuration file', required=True)
@click.option('--output', type=click.Path(), help='Output file for results (JSON format)')
def track_compare(scenarios, output):
    """Compare different track participation strategies"""
    try:
        # Load scenarios
        with open(scenarios, 'r') as f:
            scenarios_data = json.load(f)
        
        # Prepare scenarios for comparison
        comparison_scenarios = []
        for scenario in scenarios_data:
            # Create track parameters for each scenario
            track_params_list = []
            for track_config in scenario.get('tracks', []):
                track_type = track_config['track_type']
                params = track_config['parameters']
                
                if track_type == 'NODE_OPERATOR':
                    track_params = TrackParameters(
                        track_type=TrackType.NODE_OPERATOR,
                        node_operator=NodeOperatorParameters(**params)
                    )
                elif track_type == 'RISK_UNDERWRITER':
                    track_params = TrackParameters(
                        track_type=TrackType.RISK_UNDERWRITER,
                        risk_underwriter=RiskUnderwriterParameters(**params)
                    )
                elif track_type == 'LIQUIDITY_PROVIDER':
                    track_params = TrackParameters(
                        track_type=TrackType.LIQUIDITY_PROVIDER,
                        liquidity_provider=LiquidityProviderParameters(**params)
                    )
                elif track_type == 'AUCTION_PARTICIPANT':
                    track_params = TrackParameters(
                        track_type=TrackType.AUCTION_PARTICIPANT,
                        auction_participant=AuctionParticipantParameters(**params)
                    )
                track_params_list.append(track_params)
            
            # Load airdrop params if provided
            airdrop_params = None
            if 'airdrop_params' in scenario:
                airdrop_params = AirdropParameters(**scenario['airdrop_params'])
            
            comparison_scenarios.append({
                'name': scenario['name'],
                'tracks': track_params_list,
                'airdrop_params': airdrop_params
            })
        
        # Default airdrop params if not provided
        from .defaults import OPTIMIZED_DEFAULTS
        default_airdrop_params = AirdropParameters(**OPTIMIZED_DEFAULTS)
        
        calculator = AirdropCalculator(default_airdrop_params)
        comparison_result = calculator.compare_track_strategies(comparison_scenarios)
        
        # Display results
        click.echo("\nTrack Strategy Comparison")
        click.echo("=" * 60)
        click.echo(comparison_result['comparison_summary'])
        
        if comparison_result['best_scenario']:
            best = comparison_result['best_scenario']
            click.echo(f"\nBest Strategy: {best['scenario_name']}")
            click.echo(f"ROI: {best['result']['overall_roi']:.2f}%")
        
        # Save results if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(comparison_result, f, indent=2)
            click.echo(f"\nResults saved to: {output}")
            
    except (ValidationError, click.UsageError) as e:
        logger.error(f"Error: {e}")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise click.Abort()

@cli.command()
@click.option('--capital', type=float, required=True, help='Available capital for investment')
@click.option('--risk-tolerance', type=float, default=0.5, help='Risk tolerance (0.0-1.0)')
@click.option('--time-horizon', type=int, default=12, help='Investment time horizon in months')
@click.option('--min-tracks', type=int, help='Minimum number of tracks to participate in')
@click.option('--max-concentration', type=float, help='Maximum concentration in any single track')
@click.option('--target-allocation', type=float, help='Target token allocation to achieve')
@click.option('--output', type=click.Path(), help='Output file for results (JSON format)')
def optimize_tracks(capital, risk_tolerance, time_horizon, min_tracks, max_concentration,
                   target_allocation, output):
    """Find optimal track participation strategy using Z3 solver"""
    try:
        optimizer = TrackOptimizer()
        
        # Build user profile
        user_profile = {
            'available_capital': capital,
            'risk_tolerance': risk_tolerance,
            'time_horizon': time_horizon
        }
        
        # Build constraints
        constraints = {}
        if min_tracks:
            constraints['min_tracks'] = min_tracks
        if max_concentration:
            constraints['max_concentration'] = max_concentration
        
        # Run optimization
        if target_allocation:
            click.echo(f"Optimizing for target allocation: {target_allocation} tokens")
            result = optimizer.optimize_for_target_allocation(
                target_allocation, capital, risk_tolerance
            )
        else:
            click.echo("Finding optimal track weights...")
            result = optimizer.solve_optimal_track_combination(user_profile, constraints)
        
        if result.is_feasible:
            click.echo("\nOptimal Track Strategy Found!")
            click.echo("=" * 60)
            click.echo("\nTrack Weights:")
            for track_type, weight in result.track_weights.items():
                if weight > 0.01:  # Only show tracks with >1% allocation
                    click.echo(f"  {track_type.name.replace('_', ' ').title()}: {weight:.1%}")
            
            click.echo(f"\nExpected Allocation: {result.expected_allocation:,.0f} tokens")
            click.echo(f"Total Capital Required: ${result.total_capital_required:,.2f}")
            click.echo(f"Weighted Risk Score: {result.weighted_risk_score:.2f}")
            click.echo(f"Diversity Bonus: {result.diversity_bonus:.2f}")
            click.echo(f"Optimization Score: {result.optimization_score:.2f}")
            
            # Save results if output specified
            if output:
                import json
                output_data = {
                    'user_profile': user_profile,
                    'constraints': constraints,
                    'result': {
                        'track_weights': {k.name: v for k, v in result.track_weights.items()},
                        'expected_allocation': result.expected_allocation,
                        'total_capital_required': result.total_capital_required,
                        'weighted_risk_score': result.weighted_risk_score,
                        'diversity_bonus': result.diversity_bonus,
                        'optimization_score': result.optimization_score
                    }
                }
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                click.echo(f"\nResults saved to: {output}")
        else:
            click.echo("\nNo feasible solution found!")
            click.echo("Try adjusting your constraints or increasing available capital.")
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise click.Abort()

@cli.command()
@click.option('--capital', type=float, required=True, help='Available capital for investment')
@click.option('--risk-tolerance', type=float, default=0.5, help='Risk tolerance (0.0-1.0)')
@click.option('--num-solutions', type=int, default=5, help='Number of Pareto solutions to find')
@click.option('--output', type=click.Path(), help='Output file for results (JSON format)')
def pareto_tracks(capital, risk_tolerance, num_solutions, output):
    """Find Pareto-optimal track strategies"""
    try:
        optimizer = TrackOptimizer()
        
        user_profile = {
            'available_capital': capital,
            'risk_tolerance': risk_tolerance,
            'time_horizon': 12
        }
        
        click.echo(f"Finding {num_solutions} Pareto-optimal strategies...")
        solutions = optimizer.find_pareto_optimal_strategies(user_profile, num_solutions)
        
        click.echo("\nPareto-Optimal Strategies:")
        click.echo("=" * 80)
        
        for i, solution in enumerate(solutions):
            click.echo(f"\nStrategy {i+1}:")
            click.echo(f"  Allocation: {solution.expected_allocation:,.0f} tokens")
            click.echo(f"  Risk Score: {solution.weighted_risk_score:.2f}")
            click.echo(f"  Diversity: {solution.diversity_bonus:.2f}")
            
            # Show track breakdown
            active_tracks = [(t, w) for t, w in solution.track_weights.items() if w > 0.01]
            if active_tracks:
                click.echo("  Track Weights:")
                for track, weight in active_tracks:
                    click.echo(f"    {track.name.replace('_', ' ').title()}: {weight:.1%}")
        
        # Save results if output specified
        if output:
            import json
            output_data = {
                'user_profile': user_profile,
                'num_solutions': num_solutions,
                'solutions': [
                    {
                        'strategy_id': i+1,
                        'track_weights': {k.name: v for k, v in sol.track_weights.items()},
                        'expected_allocation': sol.expected_allocation,
                        'weighted_risk_score': sol.weighted_risk_score,
                        'diversity_bonus': sol.diversity_bonus,
                        'optimization_score': sol.optimization_score
                    }
                    for i, sol in enumerate(solutions)
                ]
            }
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"\nResults saved to: {output}")
            
    except Exception as e:
        logger.error(f"Error finding Pareto solutions: {e}")
        raise click.Abort()

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Path to airdrop parameters config file')
@click.option('--track-config', type=click.Path(exists=True), help='Path to track configuration file')
@click.option('--output', type=click.Path(), default='track_analysis_charts', help='Output directory for charts')
@click.option('--format', 'chart_format', type=click.Choice(['png', 'pdf', 'svg']), default='png', help='Chart format')
def track_charts(config, track_config, output, chart_format):
    """Generate comprehensive charts including track analysis"""
    try:
        # Load airdrop parameters
        if config:
            with open(config, 'r') as f:
                params_dict = json.load(f)
            params = AirdropParameters(**params_dict)
        else:
            from .defaults import OPTIMIZED_DEFAULTS
            params = AirdropParameters(**OPTIMIZED_DEFAULTS)
        
        # Initialize calculator
        calculator = AirdropCalculator(params)
        
        # Load and calculate track data if provided
        if track_config:
            with open(track_config, 'r') as f:
                track_configs = json.load(f)
            
            # Calculate track results
            track_results = []
            click.echo("Calculating track allocations...")
            
            for track_cfg in track_configs:
                track_type = track_cfg['track_type']
                if track_type == 'NODE_OPERATOR':
                    track_params = TrackParameters(
                        track_type=TrackType.NODE_OPERATOR,
                        node_operator=NodeOperatorParameters(**track_cfg['parameters'])
                    )
                elif track_type == 'RISK_UNDERWRITER':
                    track_params = TrackParameters(
                        track_type=TrackType.RISK_UNDERWRITER,
                        risk_underwriter=RiskUnderwriterParameters(**track_cfg['parameters'])
                    )
                elif track_type == 'LIQUIDITY_PROVIDER':
                    track_params = TrackParameters(
                        track_type=TrackType.LIQUIDITY_PROVIDER,
                        liquidity_provider=LiquidityProviderParameters(**track_cfg['parameters'])
                    )
                elif track_type == 'AUCTION_PARTICIPANT':
                    track_params = TrackParameters(
                        track_type=TrackType.AUCTION_PARTICIPANT,
                        auction_participant=AuctionParticipantParameters(**track_cfg['parameters'])
                    )
                
                result = calculator.calculate_track_allocation(track_params)
                track_results.append(result)
                click.echo(f"  {result['track_type']}: {result['estimated_allocation']:,.0f} tokens (ROI: {result['roi']:.1f}%)")
            
            # Store results in calculator
            calculator.track_results = track_results
        
        # Create visualizer and generate all charts
        visualizer = AirdropVisualizer(calculator)
        
        click.echo(f"\nGenerating comprehensive charts in {chart_format} format...")
        click.echo(f"Output directory: {output}")
        click.echo("-" * 60)
        
        # Generate all charts
        results = visualizer.generate_all_charts(output)
        
        click.echo("Generated chart suite:")
        for chart_type, path in results.items():
            if path:
                click.echo(f"  ✓ {chart_type.replace('_', ' ').title()}: {path}")
            else:
                click.echo(f"  ✗ {chart_type.replace('_', ' ').title()}: Failed")
        
        if track_config and 'track_dashboard' in results and results['track_dashboard']:
            click.echo("\nTrack analysis dashboard successfully generated!")
            click.echo("The dashboard includes:")
            click.echo("  • Track allocation distribution")
            click.echo("  • Points distribution by track")
            click.echo("  • Track-specific performance metrics")
            click.echo("  • Cross-track efficiency comparison")
            click.echo("  • ROI distribution analysis")
        
        click.echo("-" * 60)
        click.echo(f"Chart generation completed! Check {output} for results.")
        
    except Exception as e:
        logger.error(f"Error generating track charts: {e}")
        raise click.Abort()

if __name__ == '__main__':
    cli()
