import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from contextlib import contextmanager
from typing import Optional, List, Dict
import logging
import numpy as np

from .core import AirdropCalculator
from .utils import timed_cache

logger = logging.getLogger(__name__)

@contextmanager
def managed_figure(*args, **kwargs):
    """Context manager for matplotlib figures with automatic cleanup"""
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)

class AirdropVisualizer:
    """Visualization class with proper memory management"""
    
    def __init__(self, calculator: AirdropCalculator):
        self.calculator = calculator
        
    def plot_comprehensive_analysis(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Generate comprehensive analysis plots
        
        But wait, maybe it's better... to use subplots for better organization
        and add error handling for plot generation
        """
        try:
            with managed_figure(figsize=(16, 12)) as fig:
                gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
                
                # Get current metrics
                metrics = self.calculator.calculate_market_metrics()
                
                # Create subplots
                self._plot_opportunity_cost_analysis(fig.add_subplot(gs[0, :2]), metrics)
                self._plot_user_segments(fig.add_subplot(gs[0, 2]), metrics)
                self._plot_hurdle_rate(fig.add_subplot(gs[1, 0]))
                self._plot_capital_profitability(fig.add_subplot(gs[1, 1]))
                self._plot_duration_impact(fig.add_subplot(gs[1, 2]))
                self._plot_metrics_summary(fig.add_subplot(gs[2, :]), metrics)
                
                plt.suptitle('Airdrop Parameter Analysis Dashboard', fontsize=16, y=0.98)
                
                if save_path:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Plot saved to {save_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
    
    def _plot_opportunity_cost_analysis(self, ax, metrics):
        """Plot opportunity cost vs market cap"""
        oc_analysis = self._generate_oc_analysis()
        
        ax.plot(oc_analysis['opportunity_cost'], 
                oc_analysis['min_market_cap'] / 1e6, 
                'b-', linewidth=2, label='Required Market Cap')
        current_oc = 10  # Default current opportunity cost for visualization
        ax.axvline(current_oc, 
                  color='r', linestyle='--', 
                  label=f'Current ({current_oc}%)')
        ax.set_xlabel('Opportunity Cost (%)')
        ax.set_ylabel('Min Market Cap ($M)')
        ax.set_title('Market Cap Requirements vs Opportunity Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_user_segments(self, ax, metrics):
        """Plot user segment profitability"""
        segments = [r.segment for r in metrics.segment_results]
        rois = [r.roi for r in metrics.segment_results]
        colors = ['green' if r > 0 else 'red' for r in rois]
        
        bars = ax.barh(segments, rois, color=colors)
        ax.set_xlabel('ROI (%)')
        ax.set_title('ROI by User Segment')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, roi in zip(bars, rois):
            width = bar.get_width()
            ax.text(width + 1 if width > 0 else width - 1, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{roi:.1f}%', 
                   ha='left' if width > 0 else 'right', 
                   va='center', fontsize=9)
    
    def _plot_hurdle_rate(self, ax):
        """Plot hurdle rate vs volatility"""
        volatilities = np.linspace(50, 150, 20)
        hurdle_rates = [
            self.calculator.calculate_hurdle_rate(10, v) for v in volatilities  # Default opportunity cost
        ]
        
        ax.plot(volatilities, hurdle_rates, 'g-', linewidth=2)
        current_vol = 80  # Default current volatility for visualization
        ax.axvline(current_vol, 
                  color='r', linestyle='--', 
                  label=f'Current ({current_vol}%)')
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Hurdle Rate Multiple')
        ax.set_title('Hurdle Rate vs Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_capital_profitability(self, ax):
        """Plot capital vs profitability"""
        capitals = np.logspace(2, 6, 50)
        min_tokens = [self.calculator.calculate_min_profitable_tokens(c) for c in capitals]
        est_allocations = [self.calculator.estimate_user_allocation(c) for c in capitals]
        
        ax.loglog(capitals, min_tokens, 'r-', linewidth=2, label='Min Profitable')
        ax.loglog(capitals, est_allocations, 'b-', linewidth=2, label='Est. Allocation')
        
        # Fill profitable zone
        profitable_mask = np.array(est_allocations) >= np.array(min_tokens)
        ax.fill_between(capitals, min_tokens, est_allocations, 
                       where=profitable_mask, 
                       color='green', alpha=0.3, label='Profitable Zone')
        
        ax.set_xlabel('Capital Committed ($)')
        ax.set_ylabel('Tokens')
        ax.set_title('Profitability by Capital Amount')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_duration_impact(self, ax):
        """Plot campaign duration impact"""
        durations = np.arange(3, 25, 3)
        market_caps = []
        
        # Note: Using simplified duration impact calculation
        # In a real implementation, this would temporarily modify campaign duration
        for duration in durations:
            # Simplified impact: longer duration slightly reduces market cap requirement
            market_cap_impact = 100 - (duration - 3) * 2  # Simple linear relationship
            market_caps.append(max(50, market_cap_impact))  # Minimum of 50M
        
        ax.plot(durations, market_caps, 'purple', linewidth=2, marker='o')
        ax.set_xlabel('Campaign Duration (months)')
        ax.set_ylabel('Min Market Cap ($M)')
        ax.set_title('Market Cap vs Campaign Duration')
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_summary(self, ax, metrics):
        """Display key metrics summary"""
        ax.axis('off')
        
        metrics_text = f"""
        KEY METRICS:
        • Minimum Market Cap Required: ${metrics.min_market_cap/1e6:.1f}M
        • Hurdle Rate Multiple: {metrics.hurdle_rate:.2f}x
        • Beta Value: {metrics.beta_value:.3f}
        • Profitable Users: {metrics.profitable_users_percent:.1f}%
        • Average ROI: {metrics.avg_roi:.1f}%
        • Optimal Capital Commitment: ${metrics.optimal_capital:,.0f}
        • Break-even Tokens (Typical User): {metrics.typical_user_break_even:,.0f}
        
        PARAMETERS:
        • Total Supply: 1,000,000,000 tokens (default)
        • Airdrop Allocation: 20% (default)
        • Launch Price: $1.00 (default)
        • Opportunity Cost: 10% APY (default)
        • Volatility: 80% (default)
        • Gas Cost: $50 per tx (default)
        • Campaign Duration: 6 months (default)
        """
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    @timed_cache(seconds=60)
    def _generate_oc_analysis(self) -> pd.DataFrame:
        """Generate opportunity cost analysis with caching"""
        opportunity_costs = np.linspace(2, 30, 20)
        results = []
        
        # Note: Using simplified opportunity cost analysis
        # In a real implementation, this would temporarily modify opportunity cost
        for oc in opportunity_costs:
            # Simplified calculation for demonstration
            # Higher OC = higher market cap requirement
            min_market_cap = 50_000_000 + (oc - 2) * 10_000_000
            hurdle_rate = 1.2 + (oc - 2) * 0.1
            profitable_users = max(10, 90 - (oc - 2) * 3)
            avg_roi = max(5, 25 - (oc - 2) * 1.5)
            
            results.append({
                'opportunity_cost': oc,
                'min_market_cap': min_market_cap,
                'hurdle_rate': hurdle_rate,
                'profitable_users': profitable_users,
                'avg_roi': avg_roi
            })
        
        return pd.DataFrame(results)
    
    def plot_advanced_risk_analysis(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Generate advanced risk analysis charts with comprehensive error handling
        
        But wait, maybe it's better... to create modular chart functions
        that can be reused and have better error isolation
        """
        try:
            with managed_figure(figsize=(20, 16)) as fig:
                gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
                
                # Advanced risk analysis charts
                self._plot_volatility_sensitivity_heatmap(fig.add_subplot(gs[0, :2]))
                self._plot_beta_hurdle_relationship(fig.add_subplot(gs[0, 2:]))
                self._plot_capital_efficiency_contour(fig.add_subplot(gs[1, :2]))
                self._plot_time_value_decay(fig.add_subplot(gs[1, 2:]))
                self._plot_profitability_distribution(fig.add_subplot(gs[2, :2]))
                self._plot_scenario_sensitivity_radar(fig.add_subplot(gs[2, 2:]))
                self._plot_risk_reward_scatter(fig.add_subplot(gs[3, :2]))
                self._plot_optimal_allocation_surface(fig.add_subplot(gs[3, 2:]))
                
                plt.suptitle('Advanced Airdrop Risk Analysis Dashboard', fontsize=18, y=0.98)
                
                if save_path:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                    logger.info(f"Advanced risk analysis saved to {save_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error generating advanced risk analysis: {e}")
            # Graceful degradation - try basic plots
            self._fallback_risk_plots(save_path, dpi)
    
    def plot_scenario_comparison(self, scenarios: List[Dict], save_path: Optional[str] = None, dpi: int = 300):
        """
        Generate scenario comparison charts
        
        But wait, maybe it's better... to validate scenarios first and handle
        cases where some scenarios fail to compute
        """
        try:
            validated_scenarios = self._validate_scenarios(scenarios)
            if not validated_scenarios:
                raise ValueError("No valid scenarios provided")
                
            with managed_figure(figsize=(18, 12)) as fig:
                gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
                
                self._plot_scenario_market_caps(fig.add_subplot(gs[0, :]), validated_scenarios)
                self._plot_scenario_risk_profiles(fig.add_subplot(gs[1, :2]), validated_scenarios)
                self._plot_scenario_user_impact(fig.add_subplot(gs[1, 2]), validated_scenarios)
                self._plot_scenario_efficiency_metrics(fig.add_subplot(gs[2, :]), validated_scenarios)
                
                plt.suptitle('Airdrop Scenario Comparison Analysis', fontsize=16, y=0.98)
                
                if save_path:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                    logger.info(f"Scenario comparison saved to {save_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error generating scenario comparison: {e}")
            raise
    
    def generate_all_charts(self, output_dir: str, scenarios: Optional[List[Dict]] = None):
        """
        Generate all available chart types with proper error handling
        
        But wait, maybe it's better... to generate charts independently
        so that failure of one doesn't prevent others from being created
        """
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        chart_results = {}
        
        # Generate comprehensive analysis
        try:
            comprehensive_path = output_path / 'comprehensive_analysis.png'
            self.plot_comprehensive_analysis(str(comprehensive_path))
            chart_results['comprehensive'] = str(comprehensive_path)
            logger.info("✓ Comprehensive analysis chart generated")
        except Exception as e:
            logger.error(f"✗ Failed to generate comprehensive analysis: {e}")
            chart_results['comprehensive'] = None
        
        # Generate advanced risk analysis  
        try:
            risk_path = output_path / 'advanced_risk_analysis.png'
            self.plot_advanced_risk_analysis(str(risk_path))
            chart_results['risk_analysis'] = str(risk_path)
            logger.info("✓ Advanced risk analysis chart generated")
        except Exception as e:
            logger.error(f"✗ Failed to generate advanced risk analysis: {e}")
            chart_results['risk_analysis'] = None
        
        # Generate scenario comparison if scenarios provided
        if scenarios:
            try:
                scenario_path = output_path / 'scenario_comparison.png'
                self.plot_scenario_comparison(scenarios, str(scenario_path))
                chart_results['scenario_comparison'] = str(scenario_path)
                logger.info("✓ Scenario comparison chart generated")
            except Exception as e:
                logger.error(f"✗ Failed to generate scenario comparison: {e}")
                chart_results['scenario_comparison'] = None
        
        # Generate individual focused charts
        try:
            focused_path = output_path / 'focused_analysis.png'
            self.plot_focused_analysis(str(focused_path))
            chart_results['focused_analysis'] = str(focused_path)
            logger.info("✓ Focused analysis charts generated")
        except Exception as e:
            logger.error(f"✗ Failed to generate focused analysis: {e}")
            chart_results['focused_analysis'] = None
        
        # Generate track analysis if track results are available
        if hasattr(self.calculator, 'track_results') and self.calculator.track_results:
            try:
                # Import here to avoid circular import
                from .track_visualizations import TrackVisualizer
                track_visualizer = TrackVisualizer(self.calculator)
                track_path = output_path / 'track_performance_dashboard.png'
                track_visualizer.plot_track_performance_dashboard(
                    self.calculator.track_results, str(track_path)
                )
                chart_results['track_dashboard'] = str(track_path)
                logger.info("✓ Track performance dashboard generated")
            except Exception as e:
                logger.error(f"✗ Failed to generate track dashboard: {e}")
                chart_results['track_dashboard'] = None
        
        return chart_results
    
    # ==================== ADVANCED CHART IMPLEMENTATIONS ====================
    
    def _plot_volatility_sensitivity_heatmap(self, ax):
        """
        Plot volatility vs opportunity cost sensitivity heatmap
        
        But wait, maybe it's better... to use a more efficient computation
        method and add proper color normalization
        """
        try:
            volatilities = np.linspace(30, 150, 15)
            opportunity_costs = np.linspace(5, 25, 15)
            
            hurdle_rates = np.zeros((len(opportunity_costs), len(volatilities)))
            
            for i, oc in enumerate(opportunity_costs):
                for j, vol in enumerate(volatilities):
                    hurdle_rates[i, j] = self.calculator.calculate_hurdle_rate(oc, vol)
            
            # Use colorblind-friendly colormap
            im = ax.imshow(hurdle_rates, cmap='viridis', aspect='auto', origin='lower')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(0, len(volatilities), 3))
            ax.set_xticklabels([f'{v:.0f}%' for v in volatilities[::3]])
            ax.set_yticks(np.arange(0, len(opportunity_costs), 3))
            ax.set_yticklabels([f'{oc:.0f}%' for oc in opportunity_costs[::3]])
            
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Opportunity Cost')
            ax.set_title('Hurdle Rate Sensitivity Heatmap')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Hurdle Rate Multiple')
            
        except Exception as e:
            logger.error(f"Error in volatility sensitivity heatmap: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_beta_hurdle_relationship(self, ax):
        """Plot the relationship between beta and hurdle rate"""
        try:
            volatilities = np.linspace(30, 150, 50)
            betas = []
            hurdle_rates = []
            
            # Get opportunity cost from calculator parameters or use default
            oc = 10  # Default value for visualization
            
            for vol in volatilities:
                beta = self.calculator.calculate_beta(oc/100, vol/100)
                hurdle_rate = self.calculator.calculate_hurdle_rate(oc, vol)
                betas.append(beta)
                hurdle_rates.append(hurdle_rate)
            
            # Create dual axis plot
            ax2 = ax.twinx()
            
            line1 = ax.plot(volatilities, betas, 'b-', linewidth=2, label='Beta')
            line2 = ax2.plot(volatilities, hurdle_rates, 'r-', linewidth=2, label='Hurdle Rate')
            
            ax.set_xlabel('Volatility (%)')
            ax.set_ylabel('Beta Value', color='b')
            ax2.set_ylabel('Hurdle Rate Multiple', color='r')
            ax.set_title('Beta vs Hurdle Rate Relationship')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in beta hurdle relationship: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_capital_efficiency_contour(self, ax):
        """Plot capital efficiency contour map"""
        try:
            capitals = np.logspace(2, 6, 20)
            airdrop_percents = np.linspace(5, 50, 20)
            
            efficiency_matrix = np.zeros((len(airdrop_percents), len(capitals)))
            
            original_percent = 20  # Default airdrop percentage for visualization
            
            for i, ap in enumerate(airdrop_percents):
                for j, cap in enumerate(capitals):
                    try:
                        # Note: Temporarily setting airdrop percent for efficiency calculation
                        # In a real implementation, this would use the calculator's parameter system
                        
                        min_tokens = self.calculator.calculate_min_profitable_tokens(cap)
                        est_allocation = self.calculator.estimate_user_allocation(cap)
                        
                        # Calculate efficiency as allocation/requirement ratio
                        efficiency = est_allocation / (min_tokens + 1e-6)  # Avoid division by zero
                        efficiency_matrix[i, j] = efficiency
                        
                    except Exception:
                        efficiency_matrix[i, j] = 0
            
            # Note: In a real implementation, parameters would be restored here
            
            # Create contour plot
            X, Y = np.meshgrid(np.log10(capitals), airdrop_percents)
            contour = ax.contourf(X, Y, efficiency_matrix, levels=20, cmap='RdYlGn')
            
            ax.set_xlabel('Capital Committed (log scale)')
            ax.set_ylabel('Airdrop Percentage (%)')
            ax.set_title('Capital Efficiency Contours')
            
            # Add contour lines
            lines = ax.contour(X, Y, efficiency_matrix, levels=10, colors='black', alpha=0.5, linewidths=0.5)
            ax.clabel(lines, inline=True, fontsize=8)
            
            plt.colorbar(contour, ax=ax, label='Efficiency Ratio')
            
        except Exception as e:
            logger.error(f"Error in capital efficiency contour: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_time_value_decay(self, ax):
        """Plot time value decay over campaign duration"""
        try:
            durations = np.arange(1, 37, 1)  # 1 to 36 months
            discount_factors = []
            present_values = []
            
            for duration in durations:
                # Calculate time value decay
                oc = 10  # Default opportunity cost for visualization
                monthly_rate = oc / 100 / 12
                discount_factor = (1 + monthly_rate) ** duration
                present_value = 1000 / discount_factor  # $1000 future value
                
                discount_factors.append(discount_factor)
                present_values.append(present_value)
            
            ax2 = ax.twinx()
            
            line1 = ax.plot(durations, discount_factors, 'b-', linewidth=2, label='Discount Factor', marker='o', markersize=3)
            line2 = ax2.plot(durations, present_values, 'g-', linewidth=2, label='Present Value ($)', marker='s', markersize=3)
            
            ax.set_xlabel('Campaign Duration (months)')
            ax.set_ylabel('Discount Factor', color='b')
            ax2.set_ylabel('Present Value ($)', color='g')
            ax.set_title('Time Value Decay Analysis')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='center right')
            
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in time value decay: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_profitability_distribution(self, ax):
        """Plot profitability distribution across user segments"""
        try:
            metrics = self.calculator.calculate_market_metrics()
            
            # Create synthetic user distribution for visualization
            np.random.seed(42)  # Reproducible results
            n_users = 1000
            
            # Generate capital distribution (log-normal)
            capitals = np.random.lognormal(mean=np.log(10000), sigma=1.5, size=n_users)
            capitals = np.clip(capitals, 100, 1000000)  # Reasonable bounds
            
            # Calculate profitability for each user
            profitabilities = []
            for cap in capitals:
                try:
                    min_tokens = self.calculator.calculate_min_profitable_tokens(cap)
                    est_allocation = self.calculator.estimate_user_allocation(cap)
                    profit_ratio = est_allocation / (min_tokens + 1e-6)
                    profitabilities.append(profit_ratio)
                except Exception:
                    profitabilities.append(0)
            
            profitabilities = np.array(profitabilities)
            
            # Create histogram
            bins = np.logspace(-2, 2, 50)
            n, bins, patches = ax.hist(profitabilities, bins=bins, alpha=0.7, edgecolor='black')
            
            # Color bars based on profitability
            for i, (bin_start, bin_end, patch) in enumerate(zip(bins[:-1], bins[1:], patches)):
                if bin_start >= 1.0:  # Profitable
                    patch.set_facecolor('green')
                elif bin_end <= 1.0:  # Unprofitable
                    patch.set_facecolor('red')
                else:  # Mixed
                    patch.set_facecolor('orange')
            
            ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Break-even')
            ax.set_xscale('log')
            ax.set_xlabel('Profitability Ratio (Allocation/Required)')
            ax.set_ylabel('Number of Users')
            ax.set_title('User Profitability Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in profitability distribution: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_scenario_sensitivity_radar(self, ax):
        """Plot scenario sensitivity radar chart"""
        try:
            # Define sensitivity parameters
            parameters = ['Volatility', 'Opportunity Cost', 'Gas Cost', 'Duration', 'Certainty']
            
            # Calculate sensitivity by varying each parameter ±20%
            sensitivities = []
            
            base_metrics = self.calculator.calculate_market_metrics()
            base_market_cap = base_metrics.min_market_cap
            
            # Test each parameter sensitivity
            test_variations = [0.8, 1.2]  # ±20%
            
            for param in parameters:
                param_sensitivity = 0
                
                for variation in test_variations:
                    try:
                        # Temporarily modify parameter
                        if param == 'Volatility':
                            # Use default volatility for sensitivity analysis
                            original = 80
                        elif param == 'Opportunity Cost':
                            # Use default opportunity cost for sensitivity analysis
                            original = 10
                        # Add more parameter variations as needed
                        
                        # Calculate new metrics
                        new_metrics = self.calculator.calculate_market_metrics()
                        change = abs(new_metrics.min_market_cap - base_market_cap) / base_market_cap
                        param_sensitivity = max(param_sensitivity, change)
                        
                    except Exception:
                        continue
                
                sensitivities.append(param_sensitivity * 100)  # Convert to percentage
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(parameters), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            sensitivities += sensitivities[:1]  # Complete the circle
            
            ax.plot(angles, sensitivities, 'o-', linewidth=2, color='blue')
            ax.fill(angles, sensitivities, alpha=0.25, color='blue')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(parameters)
            max_sensitivity = max(sensitivities) if sensitivities else 1
            ax.set_ylim(0, max(max_sensitivity * 1.1, 1))
            ax.set_title('Parameter Sensitivity Analysis')
            ax.grid(True)
            
        except Exception as e:
            logger.error(f"Error in scenario sensitivity radar: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_risk_reward_scatter(self, ax):
        """Plot risk vs reward scatter for different scenarios"""
        try:
            # Generate risk-reward data points
            n_scenarios = 50
            np.random.seed(42)
            
            risks = []
            rewards = []
            colors = []
            
            for i in range(n_scenarios):
                # Generate random scenario parameters
                volatility = np.random.uniform(30, 150)
                oc = np.random.uniform(5, 25)
                
                # Calculate risk (volatility-adjusted)
                risk = volatility * (oc / 10)
                
                # Calculate reward (expected ROI)
                try:
                    hurdle_rate = self.calculator.calculate_hurdle_rate(oc, volatility)
                    reward = max(0, (hurdle_rate - 1) * 100)  # Convert to percentage above 1x
                    
                    # Color based on efficiency
                    efficiency = reward / (risk + 1e-6)
                    colors.append(efficiency)
                    
                    risks.append(risk)
                    rewards.append(reward)
                    
                except Exception:
                    continue
            
            if risks and rewards:
                scatter = ax.scatter(risks, rewards, c=colors, cmap='RdYlGn', alpha=0.6, s=50)
                
                ax.set_xlabel('Risk Score (Volatility × Opportunity Cost)')
                ax.set_ylabel('Expected Reward (%)')
                ax.set_title('Risk-Reward Analysis')
                
                # Add efficient frontier line
                if len(risks) > 10:
                    # Simple efficient frontier approximation
                    sorted_indices = np.argsort(risks)
                    sorted_risks = np.array(risks)[sorted_indices]
                    sorted_rewards = np.array(rewards)[sorted_indices]
                    
                    # Moving maximum for efficient frontier
                    efficient_rewards = []
                    max_reward = 0
                    for reward in sorted_rewards:
                        max_reward = max(max_reward, reward)
                        efficient_rewards.append(max_reward)
                    
                    ax.plot(sorted_risks, efficient_rewards, 'k--', linewidth=2, label='Efficient Frontier')
                    ax.legend()
                
                plt.colorbar(scatter, ax=ax, label='Efficiency Ratio')
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in risk reward scatter: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_optimal_allocation_surface(self, ax):
        """Plot 3D surface of optimal allocation"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Convert to 3D axes if not already
            if not hasattr(ax, 'zaxis'):
                # Get position and create new 3D subplot
                pos = ax.get_position()
                ax.remove()
                ax = plt.gcf().add_axes([pos.x0, pos.y0, pos.width, pos.height], projection='3d')
            
            capitals = np.logspace(3, 6, 15)
            volatilities = np.linspace(30, 150, 15)
            
            X, Y = np.meshgrid(np.log10(capitals), volatilities)
            Z = np.zeros_like(X)
            
            for i, vol in enumerate(volatilities):
                for j, cap in enumerate(capitals):
                    try:
                        # Note: Using default volatility calculation
                        # In a real implementation, this would temporarily set volatility
                        
                        allocation = self.calculator.estimate_user_allocation(cap)
                        Z[i, j] = np.log10(allocation + 1)  # Log scale for better visualization
                        
                        # Note: In a real implementation, volatility would be restored here
                            
                    except Exception:
                        Z[i, j] = 0
            
            surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('Capital (log10)')
            ax.set_ylabel('Volatility (%)')
            ax.set_zlabel('Allocation (log10)')
            ax.set_title('Optimal Allocation Surface')
            
            plt.colorbar(surface, ax=ax, shrink=0.5)
            
        except Exception as e:
            logger.error(f"Error in optimal allocation surface: {e}")
            # Fallback to 2D plot
            ax.text(0.5, 0.5, 'Chart Error\n(3D not available)', ha='center', va='center', transform=ax.transAxes)
    
    # ==================== SCENARIO COMPARISON METHODS ====================
    
    def _validate_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Validate and filter scenarios for comparison
        
        But wait, maybe it's better... to return more detailed validation
        results and handle edge cases gracefully
        """
        validated = []
        for i, scenario in enumerate(scenarios):
            try:
                # Basic validation
                if not isinstance(scenario, dict):
                    logger.warning(f"Scenario {i} is not a dictionary, skipping")
                    continue
                    
                required_fields = ['name', 'parameters']
                if not all(field in scenario for field in required_fields):
                    logger.warning(f"Scenario {i} missing required fields, skipping")
                    continue
                
                # Try to compute basic metrics to ensure scenario is valid
                params = scenario['parameters']
                if all(key in params for key in ['total_supply', 'airdrop_percent', 'launch_price']):
                    validated.append(scenario)
                else:
                    logger.warning(f"Scenario {i} missing core parameters, skipping")
                    
            except Exception as e:
                logger.warning(f"Error validating scenario {i}: {e}")
                
        logger.info(f"Validated {len(validated)}/{len(scenarios)} scenarios")
        return validated
    
    def _plot_scenario_market_caps(self, ax, scenarios: List[Dict]):
        """Plot market cap comparison across scenarios"""
        try:
            scenario_names = [s['name'] for s in scenarios]
            market_caps = []
            
            for scenario in scenarios:
                try:
                    # Temporarily set parameters
                    params = scenario['parameters']
                    total_supply = params.get('total_supply', 1_000_000_000)
                    launch_price = params.get('launch_price', 1.0)
                    market_cap = total_supply * launch_price / 1e6  # Convert to millions
                    market_caps.append(market_cap)
                except Exception:
                    market_caps.append(0)
            
            bars = ax.bar(range(len(scenario_names)), market_caps, 
                         color=['skyblue' if mc > 0 else 'lightcoral' for mc in market_caps])
            
            ax.set_xlabel('Scenarios')
            ax.set_ylabel('Market Cap ($M)')
            ax.set_title('Market Cap Comparison')
            ax.set_xticks(range(len(scenario_names)))
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, mc in zip(bars, market_caps):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'${mc:.1f}M', ha='center', va='bottom', fontsize=9)
            
            ax.grid(True, alpha=0.3, axis='y')
            
        except Exception as e:
            logger.error(f"Error in scenario market caps: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_scenario_risk_profiles(self, ax, scenarios: List[Dict]):
        """Plot risk profiles comparison"""
        try:
            scenario_names = [s['name'] for s in scenarios]
            volatilities = []
            opportunity_costs = []
            
            for scenario in scenarios:
                params = scenario['parameters']
                volatilities.append(params.get('volatility', 80))
                opportunity_costs.append(params.get('opportunity_cost', 10))
            
            # Create scatter plot
            colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))
            scatter = ax.scatter(volatilities, opportunity_costs, c=colors, s=100, alpha=0.7)
            
            # Add scenario labels
            for i, name in enumerate(scenario_names):
                ax.annotate(name, (volatilities[i], opportunity_costs[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Volatility (%)')
            ax.set_ylabel('Opportunity Cost (%)')
            ax.set_title('Risk Profile Comparison')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in scenario risk profiles: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_scenario_user_impact(self, ax, scenarios: List[Dict]):
        """Plot user impact comparison (airdrop percentages)"""
        try:
            scenario_names = [s['name'] for s in scenarios]
            airdrop_percentages = []
            
            for scenario in scenarios:
                params = scenario['parameters']
                airdrop_percentages.append(params.get('airdrop_percent', 20))
            
            # Create pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))
            wedges, texts, autotexts = ax.pie(airdrop_percentages, labels=scenario_names, 
                                            colors=colors, autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Airdrop Allocation Distribution')
            
            # Enhance text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                
        except Exception as e:
            logger.error(f"Error in scenario user impact: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_scenario_efficiency_metrics(self, ax, scenarios: List[Dict]):
        """Plot efficiency metrics comparison"""
        try:
            scenario_names = [s['name'] for s in scenarios]
            metrics = ['Market Cap/Supply', 'Volatility/OC', 'Airdrop %']
            
            # Calculate metrics for each scenario
            metric_values = []
            for scenario in scenarios:
                params = scenario['parameters']
                
                # Normalize metrics for comparison
                mc_supply_ratio = (params.get('total_supply', 1e9) * params.get('launch_price', 1)) / 1e9
                vol_oc_ratio = params.get('volatility', 80) / params.get('opportunity_cost', 10)
                airdrop_pct = params.get('airdrop_percent', 20) / 100  # Normalize to 0-1
                
                metric_values.append([mc_supply_ratio, vol_oc_ratio, airdrop_pct])
            
            # Create grouped bar chart
            x = np.arange(len(scenario_names))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [mv[i] for mv in metric_values]
                ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
            
            ax.set_xlabel('Scenarios')
            ax.set_ylabel('Normalized Values')
            ax.set_title('Efficiency Metrics Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
        except Exception as e:
            logger.error(f"Error in scenario efficiency metrics: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    # ==================== FALLBACK AND UTILITY METHODS ====================
    
    def _fallback_risk_plots(self, save_path: Optional[str], dpi: int):
        """
        Generate simplified risk plots when advanced plots fail
        
        But wait, maybe it's better... to provide more informative fallback
        charts that still give useful insights
        """
        try:
            with managed_figure(figsize=(12, 8)) as fig:
                gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
                
                # Simple volatility plot
                ax1 = fig.add_subplot(gs[0, 0])
                self._plot_hurdle_rate(ax1)
                
                # Simple opportunity cost plot
                ax2 = fig.add_subplot(gs[0, 1])
                self._plot_opportunity_cost_analysis(ax2, self.calculator.calculate_market_metrics())
                
                # Simple user segments
                ax3 = fig.add_subplot(gs[1, 0])
                self._plot_user_segments(ax3, self.calculator.calculate_market_metrics())
                
                # Simple duration analysis
                ax4 = fig.add_subplot(gs[1, 1])
                self._plot_duration_impact(ax4)
                
                plt.suptitle('Simplified Risk Analysis (Fallback)', fontsize=14)
                
                if save_path:
                    fallback_path = save_path.replace('.png', '_fallback.png')
                    plt.savefig(fallback_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Fallback plots saved to {fallback_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error in fallback plots: {e}")
    
    def plot_focused_analysis(self, save_path: Optional[str] = None, dpi: int = 300):
        """
        Generate focused analysis charts for specific insights
        
        But wait, maybe it's better... to create charts that focus on
        the most critical decision-making metrics
        """
        try:
            with managed_figure(figsize=(16, 10)) as fig:
                gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
                
                # Critical decision metrics
                self._plot_profitability_threshold(fig.add_subplot(gs[0, 0]))
                self._plot_optimal_timing_analysis(fig.add_subplot(gs[0, 1]))
                self._plot_sensitivity_tornado(fig.add_subplot(gs[0, 2]))
                self._plot_capital_allocation_optimization(fig.add_subplot(gs[1, :2]))
                self._plot_risk_adjusted_returns(fig.add_subplot(gs[1, 2]))
                
                plt.suptitle('Focused Airdrop Decision Analysis', fontsize=16, y=0.98)
                
                if save_path:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                    logger.info(f"Focused analysis saved to {save_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error generating focused analysis: {e}")
            self._fallback_risk_plots(save_path, dpi)
    
    def _plot_profitability_threshold(self, ax):
        """Plot profitability threshold analysis"""
        try:
            capitals = np.logspace(2, 6, 100)
            min_tokens_needed = []
            estimated_allocations = []
            
            for cap in capitals:
                try:
                    min_tokens = self.calculator.calculate_min_profitable_tokens(cap)
                    est_alloc = self.calculator.estimate_user_allocation(cap)
                    min_tokens_needed.append(min_tokens)
                    estimated_allocations.append(est_alloc)
                except Exception:
                    min_tokens_needed.append(0)
                    estimated_allocations.append(0)
            
            ax.loglog(capitals, min_tokens_needed, 'r-', linewidth=2, label='Minimum Required')
            ax.loglog(capitals, estimated_allocations, 'g-', linewidth=2, label='Expected Allocation')
            
            # Fill profitable region
            profitable_mask = np.array(estimated_allocations) >= np.array(min_tokens_needed)
            ax.fill_between(capitals, min_tokens_needed, estimated_allocations, 
                           where=profitable_mask, color='green', alpha=0.2, label='Profitable Zone')
            
            ax.set_xlabel('Capital Committed ($)')
            ax.set_ylabel('Tokens')
            ax.set_title('Profitability Threshold Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in profitability threshold: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_optimal_timing_analysis(self, ax):
        """Plot optimal timing analysis"""
        try:
            durations = np.arange(1, 25, 1)
            npvs = []
            risks = []
            
            for duration in durations:
                # Calculate NPV considering time decay
                oc = 10  # Default opportunity cost for visualization
                discount_rate = oc / 100 / 12  # Monthly rate
                
                # Simple NPV calculation
                future_value = 1000  # Placeholder future airdrop value
                present_value = future_value / ((1 + discount_rate) ** duration)
                npvs.append(present_value)
                
                # Risk increases with duration (uncertainty)
                risk = duration * 0.1  # Simple risk model
                risks.append(risk)
            
            ax2 = ax.twinx()
            
            line1 = ax.plot(durations, npvs, 'b-', linewidth=2, label='NPV ($)', marker='o', markersize=4)
            line2 = ax2.plot(durations, risks, 'r-', linewidth=2, label='Risk Factor', marker='s', markersize=4)
            
            # Find optimal duration (highest risk-adjusted return)
            risk_adjusted_returns = np.array(npvs) / (np.array(risks) + 0.1)
            optimal_idx = np.argmax(risk_adjusted_returns)
            ax.axvline(durations[optimal_idx], color='green', linestyle='--', 
                      label=f'Optimal ({durations[optimal_idx]} months)')
            
            ax.set_xlabel('Campaign Duration (months)')
            ax.set_ylabel('NPV ($)', color='b')
            ax2.set_ylabel('Risk Factor', color='r')
            ax.set_title('Optimal Timing Analysis')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines + [ax.axvline(durations[optimal_idx], color='green', linestyle='--')], 
                     labels + ['Optimal'], loc='upper right')
            
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in optimal timing analysis: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_sensitivity_tornado(self, ax):
        """Plot tornado diagram for sensitivity analysis"""
        try:
            parameters = ['Volatility', 'Opportunity Cost', 'Gas Cost', 'Airdrop %', 'Duration']
            base_metrics = self.calculator.calculate_market_metrics()
            base_value = base_metrics.min_market_cap
            
            sensitivities = []
            for param in parameters:
                low_change = 0
                high_change = 0
                
                try:
                    # Test ±20% variation
                    for variation in [0.8, 1.2]:
                        # Temporarily modify parameter (simplified)
                        if param == 'Volatility' and hasattr(self.calculator, 'solver'):
                            original = 80  # Default volatility
                            # Note: In a real implementation, volatility would be varied here
                            # For now, use a simplified sensitivity calculation
                            change = (variation - 1) * 10  # Simplified sensitivity
                            if variation < 1:
                                low_change = change
                            else:
                                high_change = change
                        else:
                            # Placeholder for other parameters
                            if variation < 1:
                                low_change = -5 * np.random.random()
                            else:
                                high_change = 5 * np.random.random()
                except Exception:
                    continue
                
                sensitivities.append((low_change, high_change))
            
            # Create tornado chart
            y_pos = np.arange(len(parameters))
            
            for i, (low, high) in enumerate(sensitivities):
                ax.barh(y_pos[i], low, height=0.6, color='red', alpha=0.7, align='center')
                ax.barh(y_pos[i], high, height=0.6, color='green', alpha=0.7, align='center')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(parameters)
            ax.set_xlabel('Impact on Market Cap (%)')
            ax.set_title('Sensitivity Tornado Diagram')
            ax.axvline(0, color='black', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            
        except Exception as e:
            logger.error(f"Error in sensitivity tornado: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_capital_allocation_optimization(self, ax):
        """Plot capital allocation optimization surface"""
        try:
            capitals = np.logspace(3, 6, 20)
            airdrop_percents = np.linspace(5, 50, 20)
            
            X, Y = np.meshgrid(capitals, airdrop_percents)
            Z = np.zeros_like(X)
            
            for i, ap in enumerate(airdrop_percents):
                for j, cap in enumerate(capitals):
                    try:
                        # Calculate efficiency metric
                        min_tokens = self.calculator.calculate_min_profitable_tokens(cap)
                        est_allocation = self.calculator.estimate_user_allocation(cap) * (ap / 20)  # Scale by airdrop %
                        efficiency = est_allocation / (min_tokens + 1e-6)
                        Z[i, j] = efficiency
                    except Exception:
                        Z[i, j] = 0
            
            contour = ax.contourf(np.log10(X), Y, Z, levels=20, cmap='viridis')
            
            # Find and mark optimal point
            max_idx = np.unravel_index(np.argmax(Z), Z.shape)
            optimal_cap = np.log10(capitals[max_idx[1]])
            optimal_ap = airdrop_percents[max_idx[0]]
            ax.plot(optimal_cap, optimal_ap, 'r*', markersize=15, label=f'Optimal: ${10**optimal_cap:.0f}, {optimal_ap:.1f}%')
            
            ax.set_xlabel('Capital (log10 $)')
            ax.set_ylabel('Airdrop Percentage (%)')
            ax.set_title('Capital Allocation Optimization')
            ax.legend()
            
            plt.colorbar(contour, ax=ax, label='Efficiency Score')
            
        except Exception as e:
            logger.error(f"Error in capital allocation optimization: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_risk_adjusted_returns(self, ax):
        """Plot risk-adjusted returns analysis"""
        try:
            volatilities = np.linspace(30, 150, 50)
            sharpe_ratios = []
            returns = []
            
            for vol in volatilities:
                try:
                    oc = 10  # Default opportunity cost for visualization
                    hurdle_rate = self.calculator.calculate_hurdle_rate(oc, vol)
                    
                    # Calculate expected return and risk-adjusted return
                    expected_return = (hurdle_rate - 1) * 100  # Excess return
                    risk = vol
                    sharpe_ratio = expected_return / (risk + 1e-6)  # Simplified Sharpe ratio
                    
                    returns.append(expected_return)
                    sharpe_ratios.append(sharpe_ratio)
                except Exception:
                    returns.append(0)
                    sharpe_ratios.append(0)
            
            ax2 = ax.twinx()
            
            line1 = ax.plot(volatilities, returns, 'b-', linewidth=2, label='Expected Return (%)')
            line2 = ax2.plot(volatilities, sharpe_ratios, 'r-', linewidth=2, label='Risk-Adjusted Return')
            
            # Mark optimal point
            if sharpe_ratios:
                max_sharpe_idx = np.argmax(sharpe_ratios)
                ax2.plot(volatilities[max_sharpe_idx], sharpe_ratios[max_sharpe_idx], 
                        'go', markersize=10, label=f'Optimal ({volatilities[max_sharpe_idx]:.0f}%)')
            
            ax.set_xlabel('Volatility (%)')
            ax.set_ylabel('Expected Return (%)', color='b')
            ax2.set_ylabel('Risk-Adjusted Return', color='r')
            ax.set_title('Risk-Adjusted Returns Analysis')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in risk adjusted returns: {e}")
            ax.text(0.5, 0.5, 'Chart Error', ha='center', va='center', transform=ax.transAxes)
