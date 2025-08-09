"""
Track-specific visualization components for the airdrop calculator
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from contextlib import contextmanager

from .types import TrackType, TrackParameters
from .tracks import MultiTrackCalculator
from .visualization import managed_figure

logger = logging.getLogger(__name__)

# Track-specific color scheme
TRACK_COLORS = {
    TrackType.NODE_OPERATOR: '#1f77b4',      # Blue
    TrackType.RISK_UNDERWRITER: '#ff7f0e',   # Orange
    TrackType.LIQUIDITY_PROVIDER: '#2ca02c',  # Green
    TrackType.AUCTION_PARTICIPANT: '#d62728'  # Red
}

TRACK_LABELS = {
    TrackType.NODE_OPERATOR: 'Node Operator',
    TrackType.RISK_UNDERWRITER: 'Risk Underwriter',
    TrackType.LIQUIDITY_PROVIDER: 'Liquidity Provider',
    TrackType.AUCTION_PARTICIPANT: 'Auction Participant'
}


class TrackVisualizer:
    """Visualization class for track-specific analysis"""
    
    def __init__(self, calculator=None):
        self.calculator = calculator
        
    def plot_track_performance_dashboard(self, track_results: List[Dict], 
                                       save_path: Optional[str] = None, dpi: int = 300):
        """Generate comprehensive track performance analysis dashboard"""
        try:
            with managed_figure(figsize=(20, 16)) as fig:
                gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
                
                # Track allocation pie chart
                self._plot_track_allocation_distribution(
                    fig.add_subplot(gs[0, :2]), track_results
                )
                
                # Points distribution by track
                self._plot_track_points_distribution(
                    fig.add_subplot(gs[0, 2:]), track_results
                )
                
                # Track-specific metrics
                self._plot_node_operator_metrics(
                    fig.add_subplot(gs[1, 0]), track_results
                )
                self._plot_risk_underwriter_metrics(
                    fig.add_subplot(gs[1, 1]), track_results
                )
                self._plot_liquidity_provider_metrics(
                    fig.add_subplot(gs[1, 2]), track_results
                )
                self._plot_auction_participant_metrics(
                    fig.add_subplot(gs[1, 3]), track_results
                )
                
                # Cross-track comparison
                self._plot_track_efficiency_comparison(
                    fig.add_subplot(gs[2, :2]), track_results
                )
                self._plot_track_profitability_matrix(
                    fig.add_subplot(gs[2, 2:]), track_results
                )
                
                # Track correlation heatmap
                self._plot_track_roi_comparison(
                    fig.add_subplot(gs[3, :]), track_results
                )
                
                plt.suptitle('Track Performance Analysis Dashboard', fontsize=16, y=0.995)
                
                if save_path:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Track performance dashboard saved to {save_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error creating track performance dashboard: {e}")
            raise
    
    def _plot_track_allocation_distribution(self, ax, track_results: List[Dict]):
        """Plot allocation distribution across tracks as a pie chart"""
        try:
            # Extract allocation data
            allocations = {}
            for result in track_results:
                track_type = result.get('track_type', 'Unknown')
                allocation = result.get('estimated_allocation', 0)
                if track_type in allocations:
                    allocations[track_type] += allocation
                else:
                    allocations[track_type] = allocation
            
            # Create pie chart
            labels = []
            sizes = []
            colors = []
            
            for track_name, allocation in allocations.items():
                labels.append(track_name.replace('_', ' ').title())
                sizes.append(allocation)
                # Get color from TRACK_COLORS if track name matches enum
                for track_type in TrackType:
                    if track_type.name == track_name:
                        colors.append(TRACK_COLORS.get(track_type, '#999999'))
                        break
                else:
                    colors.append('#999999')
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10}
            )
            
            ax.set_title('Token Allocation Distribution by Track', fontsize=12, pad=10)
            
            # Add total allocation as text
            total_allocation = sum(sizes)
            ax.text(0.5, -1.3, f'Total Allocation: {total_allocation:,.0f} tokens',
                   transform=ax.transAxes, ha='center', fontsize=10)
            
        except Exception as e:
            logger.error(f"Error plotting allocation distribution: {e}")
            ax.text(0.5, 0.5, 'Error creating pie chart', ha='center', va='center')
    
    def _plot_track_points_distribution(self, ax, track_results: List[Dict]):
        """Plot points distribution across tracks"""
        try:
            # Prepare data
            tracks = []
            points = []
            colors = []
            
            for result in track_results:
                track_name = result.get('track_type', 'Unknown')
                track_points = result.get('points', 0)
                
                tracks.append(track_name.replace('_', ' ').title())
                points.append(track_points)
                
                # Get color
                for track_type in TrackType:
                    if track_type.name == track_name:
                        colors.append(TRACK_COLORS.get(track_type, '#999999'))
                        break
                else:
                    colors.append('#999999')
            
            # Create bar chart
            bars = ax.bar(tracks, points, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar, point in zip(bars, points):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{point:,.0f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Track', fontsize=11)
            ax.set_ylabel('Points Earned', fontsize=11)
            ax.set_title('Points Distribution by Track', fontsize=12, pad=10)
            
            # Rotate x labels if needed
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add grid
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            
        except Exception as e:
            logger.error(f"Error plotting points distribution: {e}")
            ax.text(0.5, 0.5, 'Error creating bar chart', ha='center', va='center')
    
    def _plot_node_operator_metrics(self, ax, track_results: List[Dict]):
        """Visualize node operator specific metrics"""
        try:
            # Filter for node operator results
            node_results = [r for r in track_results 
                          if r.get('track_type') == TrackType.NODE_OPERATOR.name]
            
            if not node_results:
                ax.text(0.5, 0.5, 'No Node Operator Data', ha='center', va='center')
                ax.set_title('Node Operator Metrics', fontsize=11)
                return
            
            # Example: Create a scatter plot of capital vs allocation
            capitals = [r.get('capital_equivalent', 0) for r in node_results]
            allocations = [r.get('estimated_allocation', 0) for r in node_results]
            rois = [r.get('roi', -100) for r in node_results]
            
            # Color by ROI
            scatter = ax.scatter(capitals, allocations, c=rois, cmap='RdYlGn',
                               s=100, alpha=0.7, edgecolors='black')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('ROI (%)', fontsize=9)
            
            ax.set_xlabel('Capital Equivalent ($)', fontsize=10)
            ax.set_ylabel('Token Allocation', fontsize=10)
            ax.set_title('Node Operator: Capital vs Allocation', fontsize=11)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting node operator metrics: {e}")
            ax.text(0.5, 0.5, 'Error creating plot', ha='center', va='center')
    
    def _plot_risk_underwriter_metrics(self, ax, track_results: List[Dict]):
        """Visualize risk underwriter staking patterns"""
        try:
            # Filter for risk underwriter results
            risk_results = [r for r in track_results 
                          if r.get('track_type') == TrackType.RISK_UNDERWRITER.name]
            
            if not risk_results:
                ax.text(0.5, 0.5, 'No Risk Underwriter Data', ha='center', va='center')
                ax.set_title('Risk Underwriter Metrics', fontsize=11)
                return
            
            # Example: Bar chart of staking returns
            labels = [f"Stake {i+1}" for i in range(len(risk_results))]
            rois = [r.get('roi', -100) for r in risk_results]
            
            colors = ['green' if roi > 0 else 'red' for roi in rois]
            bars = ax.bar(labels, rois, color=colors, alpha=0.7, edgecolor='black')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels
            for bar, roi in zip(bars, rois):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{roi:.1f}%', ha='center', 
                       va='bottom' if roi > 0 else 'top', fontsize=9)
            
            ax.set_xlabel('Staking Position', fontsize=10)
            ax.set_ylabel('ROI (%)', fontsize=10)
            ax.set_title('Risk Underwriter: Staking Returns', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting risk underwriter metrics: {e}")
            ax.text(0.5, 0.5, 'Error creating plot', ha='center', va='center')
    
    def _plot_liquidity_provider_metrics(self, ax, track_results: List[Dict]):
        """Visualize liquidity provision patterns"""
        try:
            # Filter for LP results
            lp_results = [r for r in track_results 
                         if r.get('track_type') == TrackType.LIQUIDITY_PROVIDER.name]
            
            if not lp_results:
                ax.text(0.5, 0.5, 'No Liquidity Provider Data', ha='center', va='center')
                ax.set_title('Liquidity Provider Metrics', fontsize=11)
                return
            
            # Example: Efficiency plot
            capitals = [r.get('capital_equivalent', 1) for r in lp_results]
            allocations = [r.get('estimated_allocation', 0) for r in lp_results]
            efficiency = [a/c if c > 0 else 0 for a, c in zip(allocations, capitals)]
            
            ax.bar(range(len(efficiency)), efficiency, 
                  color=TRACK_COLORS[TrackType.LIQUIDITY_PROVIDER],
                  alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('LP Position', fontsize=10)
            ax.set_ylabel('Tokens per Dollar', fontsize=10)
            ax.set_title('Liquidity Provider: Capital Efficiency', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting LP metrics: {e}")
            ax.text(0.5, 0.5, 'Error creating plot', ha='center', va='center')
    
    def _plot_auction_participant_metrics(self, ax, track_results: List[Dict]):
        """Visualize auction participation effectiveness"""
        try:
            # Filter for auction results
            auction_results = [r for r in track_results 
                             if r.get('track_type') == TrackType.AUCTION_PARTICIPANT.name]
            
            if not auction_results:
                ax.text(0.5, 0.5, 'No Auction Participant Data', ha='center', va='center')
                ax.set_title('Auction Participant Metrics', fontsize=11)
                return
            
            # Example: Success vs ROI scatter
            points = [r.get('points', 0) for r in auction_results]
            rois = [r.get('roi', -100) for r in auction_results]
            
            ax.scatter(points, rois, 
                      color=TRACK_COLORS[TrackType.AUCTION_PARTICIPANT],
                      s=100, alpha=0.7, edgecolors='black')
            
            # Add trend line if enough data
            if len(points) > 1:
                z = np.polyfit(points, rois, 1)
                p = np.poly1d(z)
                ax.plot(sorted(points), p(sorted(points)), "r--", alpha=0.8)
            
            ax.set_xlabel('Points Earned', fontsize=10)
            ax.set_ylabel('ROI (%)', fontsize=10)
            ax.set_title('Auction Participant: Points vs ROI', fontsize=11)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting auction metrics: {e}")
            ax.text(0.5, 0.5, 'Error creating plot', ha='center', va='center')
    
    def _plot_track_efficiency_comparison(self, ax, track_results: List[Dict]):
        """Compare efficiency across tracks"""
        try:
            # Calculate efficiency metrics by track
            track_efficiency = {}
            
            for result in track_results:
                track_name = result.get('track_type', 'Unknown')
                capital = result.get('capital_equivalent', 1)
                allocation = result.get('estimated_allocation', 0)
                
                if capital > 0:
                    efficiency = allocation / capital
                    if track_name in track_efficiency:
                        track_efficiency[track_name].append(efficiency)
                    else:
                        track_efficiency[track_name] = [efficiency]
            
            # Calculate average efficiency per track
            tracks = []
            avg_efficiency = []
            colors = []
            
            for track_name, efficiencies in track_efficiency.items():
                tracks.append(track_name.replace('_', ' ').title())
                avg_efficiency.append(np.mean(efficiencies))
                
                # Get color
                for track_type in TrackType:
                    if track_type.name == track_name:
                        colors.append(TRACK_COLORS.get(track_type, '#999999'))
                        break
                else:
                    colors.append('#999999')
            
            # Create horizontal bar chart
            bars = ax.barh(tracks, avg_efficiency, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, eff in zip(bars, avg_efficiency):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{eff:.2f}', ha='left', va='center', fontsize=9)
            
            ax.set_xlabel('Average Tokens per Dollar', fontsize=11)
            ax.set_title('Track Efficiency Comparison', fontsize=12, pad=10)
            ax.grid(axis='x', alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting efficiency comparison: {e}")
            ax.text(0.5, 0.5, 'Error creating comparison', ha='center', va='center')
    
    def _plot_track_profitability_matrix(self, ax, track_results: List[Dict]):
        """Matrix visualization of track profitability"""
        try:
            # Create profitability matrix
            track_names = [t.name for t in TrackType]
            matrix_data = np.zeros((len(track_names), 3))  # 3 metrics: ROI, Allocation, Risk
            
            # Fill matrix with average values
            for i, track_type in enumerate(TrackType):
                track_data = [r for r in track_results 
                            if r.get('track_type') == track_type.name]
                
                if track_data:
                    avg_roi = np.mean([r.get('roi', -100) for r in track_data])
                    avg_allocation = np.mean([r.get('estimated_allocation', 0) for r in track_data])
                    avg_risk = np.mean([r.get('risk_factor', 1.0) for r in track_data])
                    
                    matrix_data[i, 0] = avg_roi
                    matrix_data[i, 1] = avg_allocation / 1000  # Scale down for visualization
                    matrix_data[i, 2] = avg_risk
            
            # Create heatmap
            metrics = ['Avg ROI (%)', 'Avg Allocation (K)', 'Risk Factor']
            track_labels = [TRACK_LABELS[t] for t in TrackType]
            
            sns.heatmap(matrix_data.T, 
                       xticklabels=track_labels,
                       yticklabels=metrics,
                       annot=True, fmt='.1f',
                       cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Value'},
                       ax=ax)
            
            ax.set_title('Track Profitability Matrix', fontsize=12, pad=10)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        except Exception as e:
            logger.error(f"Error plotting profitability matrix: {e}")
            ax.text(0.5, 0.5, 'Error creating matrix', ha='center', va='center')
    
    def _plot_track_roi_comparison(self, ax, track_results: List[Dict]):
        """Compare ROI across all tracks"""
        try:
            # Prepare data for box plot
            roi_by_track = {}
            
            for result in track_results:
                track_name = result.get('track_type', 'Unknown')
                roi = result.get('roi', -100)
                
                if track_name in roi_by_track:
                    roi_by_track[track_name].append(roi)
                else:
                    roi_by_track[track_name] = [roi]
            
            # Create box plot
            tracks = []
            roi_data = []
            colors = []
            
            for track_type in TrackType:
                if track_type.name in roi_by_track:
                    tracks.append(TRACK_LABELS[track_type])
                    roi_data.append(roi_by_track[track_type.name])
                    colors.append(TRACK_COLORS[track_type])
            
            if roi_data:
                bp = ax.boxplot(roi_data, labels=tracks, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add zero line
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                
                ax.set_ylabel('ROI (%)', fontsize=11)
                ax.set_title('ROI Distribution by Track', fontsize=12, pad=10)
                ax.grid(axis='y', alpha=0.3)
                
                # Rotate x labels if needed
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, 'No ROI data available', ha='center', va='center')
            
        except Exception as e:
            logger.error(f"Error plotting ROI comparison: {e}")
            ax.text(0.5, 0.5, 'Error creating comparison', ha='center', va='center')
    
    def plot_track_sensitivity_spider(self, track_params: TrackParameters,
                                    save_path: Optional[str] = None, dpi: int = 300):
        """Spider/radar chart for track parameter sensitivity"""
        try:
            with managed_figure(figsize=(10, 10)) as fig:
                ax = fig.add_subplot(111, projection='polar')
                
                # Define parameters to analyze
                categories = ['Duration Impact', 'Capital Sensitivity', 
                            'Performance Factor', 'Risk Level', 'ROI Potential']
                
                # Calculate sensitivity scores (example values)
                # In real implementation, these would be calculated from actual data
                values = {
                    TrackType.NODE_OPERATOR: [0.8, 0.9, 0.7, 0.6, 0.5],
                    TrackType.RISK_UNDERWRITER: [0.9, 0.7, 0.5, 0.8, 0.6],
                    TrackType.LIQUIDITY_PROVIDER: [0.7, 0.6, 0.4, 0.5, 0.8],
                    TrackType.AUCTION_PARTICIPANT: [0.5, 0.5, 0.9, 0.7, 0.7]
                }
                
                # Number of variables
                num_vars = len(categories)
                
                # Compute angle for each axis
                angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
                angles += angles[:1]
                
                # Plot for each track
                for track_type, color in TRACK_COLORS.items():
                    if track_type in values:
                        vals = values[track_type]
                        vals += vals[:1]  # Complete the circle
                        
                        ax.plot(angles, vals, 'o-', linewidth=2, 
                               label=TRACK_LABELS[track_type], color=color)
                        ax.fill(angles, vals, alpha=0.25, color=color)
                
                # Fix axis to go in the right order and start at 12 o'clock
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw axis labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                
                # Set y-axis limits
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                ax.set_yticklabels(['20%', '40%', '60%', '80%'])
                
                # Add legend
                plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                
                plt.title('Track Sensitivity Analysis', size=14, pad=20)
                
                if save_path:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Track sensitivity spider saved to {save_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error creating sensitivity spider chart: {e}")
            raise
    
    def plot_track_evolution_timeline(self, time_periods: List[int], 
                                    track_data: Dict[int, List[Dict]],
                                    save_path: Optional[str] = None, dpi: int = 300):
        """Animated or multi-panel timeline showing track metrics evolution"""
        try:
            with managed_figure(figsize=(16, 10)) as fig:
                gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
                
                # Plot evolution for each track type
                for i, track_type in enumerate(TrackType):
                    ax = fig.add_subplot(gs[i // 2, i % 2])
                    
                    # Extract time series data for this track
                    roi_series = []
                    allocation_series = []
                    
                    for period in time_periods:
                        period_data = track_data.get(period, [])
                        track_period_data = [d for d in period_data 
                                           if d.get('track_type') == track_type.name]
                        
                        if track_period_data:
                            avg_roi = np.mean([d.get('roi', -100) for d in track_period_data])
                            avg_allocation = np.mean([d.get('estimated_allocation', 0) 
                                                    for d in track_period_data])
                        else:
                            avg_roi = 0
                            avg_allocation = 0
                        
                        roi_series.append(avg_roi)
                        allocation_series.append(avg_allocation)
                    
                    # Plot dual axis
                    color = TRACK_COLORS[track_type]
                    
                    ax.plot(time_periods, roi_series, color=color, marker='o',
                           linewidth=2, label='ROI %')
                    ax.set_xlabel('Time Period (months)')
                    ax.set_ylabel('ROI (%)', color=color)
                    ax.tick_params(axis='y', labelcolor=color)
                    
                    # Create second y-axis
                    ax2 = ax.twinx()
                    ax2.plot(time_periods, allocation_series, color='gray', 
                            marker='s', linewidth=2, linestyle='--', 
                            label='Allocation')
                    ax2.set_ylabel('Token Allocation', color='gray')
                    ax2.tick_params(axis='y', labelcolor='gray')
                    
                    ax.set_title(f'{TRACK_LABELS[track_type]} Evolution', fontsize=11)
                    ax.grid(True, alpha=0.3)
                    
                    # Add zero line for ROI
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                
                plt.suptitle('Track Performance Evolution Over Time', fontsize=14)
                
                if save_path:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Track evolution timeline saved to {save_path}")
                else:
                    plt.show()
                    
        except Exception as e:
            logger.error(f"Error creating evolution timeline: {e}")
            raise


def create_track_comparison_report(track_results: List[Dict], 
                                 output_path: str = "track_comparison_report.png"):
    """Create a comprehensive track comparison report"""
    visualizer = TrackVisualizer()
    visualizer.plot_track_performance_dashboard(track_results, save_path=output_path)
    return output_path