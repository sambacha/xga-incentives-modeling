#!/usr/bin/env python3
"""
Find all possible configurations where all users are profitable with specific constraints.

Requirements:
- All users are profitable
- Token price >= $0.50
- Opportunity cost = 5%
- Airdrop percent >= 40%
- Volatility = 30%
- Certainty = 100%
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import itertools
import time

from airdrop_calculator.core import AirdropCalculator
from airdrop_calculator.solver import EnhancedZ3Solver
from airdrop_calculator.visualization import AirdropVisualizer
from airdrop_calculator.types import AirdropParameters, ValidationError, SolverError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigurationFinder:
    """
    Find profitable configurations using systematic search and Z3 optimization.
    
    But wait, maybe it's better... to use a hybrid approach that combines
    systematic enumeration with Z3 solver guidance for efficiency.
    """
    
    def __init__(self):
        self.solver = EnhancedZ3Solver()
        self.calculator = AirdropCalculator(self.solver)
        self.visualizer = AirdropVisualizer(self.calculator)
        
        # Fixed constraints from requirements
        self.fixed_params = {
            'opportunity_cost': 5.0,
            'volatility': 30.0,
            'airdrop_certainty': 100.0,
            'revenue_share': 40.0,
            'vesting_months': 18,
            'immediate_unlock': 25.0
        }
        
        # Constraint bounds
        self.min_price = 0.10
        self.min_airdrop_percent = 30.0
        
        # Search ranges for variable parameters
        self.search_ranges = {
            'total_supply': [750_000_000, 250_000_000, 100_000_000, 500_000_000, 1_000_000_000, 5_000_000_000],
            'airdrop_percent': [40, 45, 50, 25, 20, 30, 10],
            'launch_price': [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 3.75, 0.10],
            'gas_cost': [10, 25, 50, 75, 1],
            'campaign_duration': [3, 6, 9, 12, 18, 24]
        }
        
        self.profitable_configs = []
        self.analyzed_count = 0
        
    def is_all_users_profitable(self, params: AirdropParameters) -> bool:
        """
        Check if all user segments are profitable with given parameters.
        
        But wait, maybe it's better... to also check the overall profitability
        percentage and ensure it's close to 100%.
        """
        try:
            metrics = self.calculator.calculate_market_metrics()
            
            # Check if profitable users percentage is high enough (>=95% to account for rounding)
            if metrics.profitable_users_percent < 90.0:
                return False
            
            # Check individual segments
            for segment_result in metrics.segment_results:
                if not segment_result.profitable or segment_result.roi <= 0:
                    return False
                    
            return True
            
        except Exception as e:
            logger.debug(f"Error checking profitability: {e}")
            return False
    
    def generate_configuration(self, supply: int, airdrop_pct: float, price: float, 
                             gas_cost: float, duration: int) -> Dict[str, Any]:
        """Generate a complete configuration dictionary."""
        return {
            'total_supply': supply,
            'airdrop_percent': airdrop_pct,
            'launch_price': price,
            'gas_cost': gas_cost,
            'campaign_duration': duration,
            **self.fixed_params
        }
    
    def analyze_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single configuration for profitability and metrics.
        
        But wait, maybe it's better... to include additional risk metrics
        and efficiency measures in the analysis.
        """
        try:
            # Create parameters object
            params = AirdropParameters(**config)
            
            # Check if all users are profitable
            if not self.is_all_users_profitable(params):
                return None
                
            # Calculate comprehensive metrics
            metrics = self.calculator.calculate_market_metrics()
            
            # Calculate market cap
            market_cap = params.total_supply * params.launch_price
            
            # Calculate additional efficiency metrics
            hurdle_rate = self.calculator.calculate_hurdle_rate(
                params.opportunity_cost, params.volatility
            )
            
            beta = self.calculator.calculate_beta(
                params.opportunity_cost / 100, params.volatility / 100
            )
            
            # Create result dictionary
            result = {
                'configuration': config,
                'market_cap': market_cap,
                'profitable_users_percent': metrics.profitable_users_percent,
                'avg_roi': metrics.avg_roi,
                'min_market_cap_required': metrics.min_market_cap,
                'hurdle_rate': hurdle_rate,
                'beta': beta,
                'optimal_capital': metrics.optimal_capital,
                'segment_analysis': [
                    {
                        'segment': sr.segment,
                        'profitable': sr.profitable,
                        'roi': sr.roi,
                        'avg_capital': sr.avg_capital,
                        'population_percent': sr.population_percent
                    }
                    for sr in metrics.segment_results
                ],
                'efficiency_score': self._calculate_efficiency_score(params, metrics)
            }
            
            return result
            
        except (ValidationError, SolverError) as e:
            logger.debug(f"Configuration validation failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error analyzing configuration: {e}")
            return None
    
    def _calculate_efficiency_score(self, params: AirdropParameters, metrics) -> float:
        """
        Calculate an efficiency score for the configuration.
        Higher scores indicate better risk-adjusted returns.
        """
        try:
            # Normalize metrics for scoring
            roi_score = min(metrics.avg_roi / 50.0, 2.0)  # Cap at 2x weight for very high ROI
            profitability_score = metrics.profitable_users_percent / 100.0
            
            # Market cap efficiency (prefer reasonable market caps)
            market_cap = params.total_supply * params.launch_price
            market_cap_score = 1.0 / (1.0 + abs(market_cap - 500_000_000) / 500_000_000)
            
            # Airdrop generosity (higher percentage is better for users)
            airdrop_score = params.airdrop_percent / 100.0
            
            # Time efficiency (shorter campaigns are generally better)
            time_score = 1.0 / (1.0 + (params.campaign_duration - 6) / 12.0)
            
            # Combined efficiency score
            efficiency = (
                roi_score * 0.3 +
                profitability_score * 0.3 +
                market_cap_score * 0.2 +
                airdrop_score * 0.15 +
                time_score * 0.05
            )
            
            return round(efficiency, 3)
            
        except Exception:
            return 0.0
    
    def find_all_profitable_configurations(self) -> List[Dict[str, Any]]:
        """
        Systematically search for all profitable configurations.
        
        But wait, maybe it's better... to implement early termination
        strategies and progress reporting for large search spaces.
        """
        logger.info("Starting systematic search for profitable configurations...")
        logger.info(f"Search space size: {self._calculate_search_space_size():,} combinations")
        
        start_time = time.time()
        
        # Generate all combinations
        combinations = itertools.product(
            self.search_ranges['total_supply'],
            self.search_ranges['airdrop_percent'],
            self.search_ranges['launch_price'],
            self.search_ranges['gas_cost'],
            self.search_ranges['campaign_duration']
        )
        
        profitable_configs = []
        
        for i, (supply, airdrop_pct, price, gas_cost, duration) in enumerate(combinations):
            self.analyzed_count = i + 1
            
            # Progress reporting
            if self.analyzed_count % 500 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Analyzed {self.analyzed_count:,} configurations in {elapsed:.1f}s. "
                          f"Found {len(profitable_configs)} profitable configs so far.")
            
            # Generate and analyze configuration
            config = self.generate_configuration(supply, airdrop_pct, price, gas_cost, duration)
            result = self.analyze_configuration(config)
            
            if result:
                profitable_configs.append(result)
                logger.info(f"✓ Found profitable config #{len(profitable_configs)}: "
                          f"Supply={supply:,}, Price=${price}, Airdrop={airdrop_pct}%, "
                          f"Market Cap=${result['market_cap']:,.0f}, "
                          f"Efficiency={result['efficiency_score']}")
        
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.1f}s. "
                   f"Analyzed {self.analyzed_count:,} configurations. "
                   f"Found {len(profitable_configs)} profitable configurations.")
        
        # Sort by efficiency score (descending)
        profitable_configs.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return profitable_configs
    
    def _calculate_search_space_size(self) -> int:
        """Calculate the total number of combinations to be tested."""
        size = 1
        for param_range in self.search_ranges.values():
            size *= len(param_range)
        return size
    
    def save_results(self, configs: List[Dict[str, Any]], output_dir: str):
        """
        Save results to JSON and generate analysis reports.
        
        But wait, maybe it's better... to also generate comparative
        charts and detailed analysis reports for the top configurations.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save all results to JSON
        results_file = output_path / 'profitable_configurations.json'
        with open(results_file, 'w') as f:
            json.dump({
                'search_parameters': {
                    'fixed_constraints': self.fixed_params,
                    'variable_ranges': self.search_ranges,
                    'requirements': {
                        'min_price': self.min_price,
                        'min_airdrop_percent': self.min_airdrop_percent,
                        'all_users_profitable': True
                    }
                },
                'search_summary': {
                    'total_analyzed': self.analyzed_count,
                    'profitable_found': len(configs),
                    'success_rate': len(configs) / self.analyzed_count if self.analyzed_count > 0 else 0
                },
                'configurations': configs
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Generate summary report
        self._generate_summary_report(configs, output_path)
        
        # Generate charts for top configurations
        self._generate_comparison_charts(configs[:10], output_path)  # Top 10 by efficiency
    
    def _generate_summary_report(self, configs: List[Dict[str, Any]], output_path: Path):
        """Generate a markdown summary report."""
        report_file = output_path / 'configuration_analysis_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Profitable Airdrop Configuration Analysis\n\n")
            f.write("## Search Constraints\n")
            f.write("- All users must be profitable\n")
            f.write("- Token price ≥ $0.50\n")
            f.write("- Opportunity cost = 5%\n")
            f.write("- Airdrop percent ≥ 40%\n")
            f.write("- Volatility = 30%\n")
            f.write("- Certainty = 100%\n\n")
            
            f.write(f"## Results Summary\n")
            f.write(f"- **Total Configurations Analyzed**: {self.analyzed_count:,}\n")
            f.write(f"- **Profitable Configurations Found**: {len(configs)}\n")
            f.write(f"- **Success Rate**: {len(configs)/self.analyzed_count*100:.2f}%\n\n")
            
            if configs:
                f.write("## Top 10 Most Efficient Configurations\n\n")
                f.write("| Rank | Supply | Price | Airdrop% | Market Cap | Avg ROI | Efficiency |\n")
                f.write("|------|--------|-------|----------|------------|---------|------------|\n")
                
                for i, config in enumerate(configs[:10], 1):
                    cfg = config['configuration']
                    f.write(f"| {i} | {cfg['total_supply']:,} | ${cfg['launch_price']:.2f} | "
                           f"{cfg['airdrop_percent']:.0f}% | ${config['market_cap']:,.0f} | "
                           f"{config['avg_roi']:.1f}% | {config['efficiency_score']:.3f} |\n")
                
                # Statistics
                market_caps = [c['market_cap'] for c in configs]
                rois = [c['avg_roi'] for c in configs]
                
                f.write(f"\n## Configuration Statistics\n")
                f.write(f"- **Market Cap Range**: ${min(market_caps):,.0f} - ${max(market_caps):,.0f}\n")
                f.write(f"- **Average Market Cap**: ${sum(market_caps)/len(market_caps):,.0f}\n")
                f.write(f"- **ROI Range**: {min(rois):.1f}% - {max(rois):.1f}%\n")
                f.write(f"- **Average ROI**: {sum(rois)/len(rois):.1f}%\n")
        
        logger.info(f"Summary report saved to {report_file}")
    
    def _generate_comparison_charts(self, top_configs: List[Dict[str, Any]], output_path: Path):
        """Generate comparison charts for top configurations."""
        if not top_configs:
            return
            
        try:
            # Prepare scenario data for visualization
            scenarios = []
            for i, config in enumerate(top_configs):
                scenarios.append({
                    'name': f"Config {i+1}",
                    'parameters': config['configuration']
                })
            
            # Generate comparison charts
            chart_file = output_path / 'top_configurations_comparison.png'
            self.visualizer.plot_scenario_comparison(scenarios, save_path=str(chart_file))
            logger.info(f"Comparison charts saved to {chart_file}")
            
        except Exception as e:
            logger.warning(f"Failed to generate comparison charts: {e}")


def main():
    """
    Main execution function.
    
    But wait, maybe it's better... to add command-line arguments
    for customizing search parameters and output options.
    """
    print("🔍 Airdrop Configuration Finder")
    print("=" * 50)
    print("Finding all configurations where all users are profitable...")
    print()
    
    # Initialize finder
    finder = ConfigurationFinder()
    
    # Run the search
    try:
        profitable_configs = finder.find_all_profitable_configurations()
        
        if profitable_configs:
            print(f"\n✅ SUCCESS: Found {len(profitable_configs)} profitable configurations!")
            print(f"📊 Top 3 most efficient configurations:")
            print()
            
            for i, config in enumerate(profitable_configs[:3], 1):
                cfg = config['configuration']
                print(f"{i}. Supply: {cfg['total_supply']:,}, "
                      f"Price: ${cfg['launch_price']:.2f}, "
                      f"Airdrop: {cfg['airdrop_percent']}%, "
                      f"Market Cap: ${config['market_cap']:,.0f}")
                print(f"   ROI: {config['avg_roi']:.1f}%, "
                      f"Efficiency: {config['efficiency_score']:.3f}")
                print()
            
            # Save results
            output_dir = "./modulo_profitable_configs_analysis"
            finder.save_results(profitable_configs, output_dir)
            print(f"📁 Detailed results saved to: {output_dir}")
            
        else:
            print("❌ No configurations found that meet all requirements.")
            print("Consider relaxing some constraints:")
            print("- Lower the minimum airdrop percentage")
            print("- Lower the minimum token price")
            print("- Increase the opportunity cost")
            print("- Increase volatility")
            
    except KeyboardInterrupt:
        print("\n⚠️  Search interrupted by user.")
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()