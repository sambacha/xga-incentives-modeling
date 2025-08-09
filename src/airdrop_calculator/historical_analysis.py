"""
Historical analysis framework for backtesting and performance tracking
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

from .types import AirdropParameters, TrackType, TrackParameters
from .core import AirdropCalculator
from .tracks import MultiTrackCalculator
from .defaults import OPTIMIZED_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a single simulation run"""
    timestamp: str
    parameters: Dict[str, Any]
    track_results: Dict[str, Any]
    actual_roi: float
    success: bool
    market_conditions: Dict[str, float]
    notes: Optional[str] = None


@dataclass
class BacktestResult:
    """Result of backtesting a strategy"""
    strategy_name: str
    total_simulations: int
    successful_simulations: int
    average_roi: float
    std_roi: float
    sharpe_ratio: float
    max_drawdown: float
    best_market_conditions: Dict[str, float]
    worst_market_conditions: Dict[str, float]


class HistoricalAnalyzer:
    """Analyze historical simulation data and backtest strategies"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else Path("data/historical_simulations.json")
        self.simulations: List[SimulationResult] = []
        self.df: Optional[pd.DataFrame] = None
        
    def load_historical_data(self, filepath: Optional[str] = None) -> int:
        """Load historical simulation data from file"""
        load_path = Path(filepath) if filepath else self.data_path
        
        if not load_path.exists():
            logger.warning(f"Historical data file not found: {load_path}")
            return 0
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Convert to SimulationResult objects
            self.simulations = []
            for sim_data in data.get('simulations', []):
                self.simulations.append(SimulationResult(**sim_data))
            
            # Create DataFrame for easier analysis
            self._create_dataframe()
            
            logger.info(f"Loaded {len(self.simulations)} historical simulations")
            return len(self.simulations)
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return 0
    
    def save_simulation(self, result: SimulationResult, filepath: Optional[str] = None):
        """Save a new simulation result to historical data"""
        save_path = Path(filepath) if filepath else self.data_path
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        existing_data = {'simulations': []}
        if save_path.exists():
            try:
                with open(save_path, 'r') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Add new simulation
        existing_data['simulations'].append(asdict(result))
        
        # Save updated data
        with open(save_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        # Reload data
        self.load_historical_data(str(save_path))
    
    def _create_dataframe(self):
        """Create pandas DataFrame from simulations for analysis"""
        if not self.simulations:
            self.df = pd.DataFrame()
            return
        
        # Extract data for DataFrame
        data = []
        for sim in self.simulations:
            row = {
                'timestamp': pd.to_datetime(sim.timestamp),
                'actual_roi': sim.actual_roi,
                'success': sim.success,
                'total_supply': sim.parameters.get('total_supply', 0),
                'airdrop_percent': sim.parameters.get('airdrop_percent', 0),
                'launch_price': sim.parameters.get('launch_price', 0),
                'opportunity_cost': sim.parameters.get('opportunity_cost', 0),
                'volatility': sim.parameters.get('volatility', 0),
                'gas_cost': sim.parameters.get('gas_cost', 0),
            }
            
            # Add market conditions
            for key, value in sim.market_conditions.items():
                row[f'market_{key}'] = value
            
            # Add track data if available
            if sim.track_results:
                for track_name, track_data in sim.track_results.items():
                    if isinstance(track_data, dict):
                        row[f'{track_name}_roi'] = track_data.get('roi', 0)
                        row[f'{track_name}_allocation'] = track_data.get('allocation', 0)
            
            data.append(row)
        
        self.df = pd.DataFrame(data)
        self.df.set_index('timestamp', inplace=True)
        self.df.sort_index(inplace=True)
    
    def backtest_strategy(self, strategy: Dict[str, Any], 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> BacktestResult:
        """Backtest a strategy against historical data"""
        if self.df is None or self.df.empty:
            logger.warning("No historical data loaded for backtesting")
            return BacktestResult(
                strategy_name=strategy.get('name', 'Unknown'),
                total_simulations=0,
                successful_simulations=0,
                average_roi=0,
                std_roi=0,
                sharpe_ratio=0,
                max_drawdown=0,
                best_market_conditions={},
                worst_market_conditions={}
            )
        
        # Filter by date range if specified
        df = self.df
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Apply strategy filters
        if 'parameter_ranges' in strategy:
            for param, range_dict in strategy['parameter_ranges'].items():
                if param in df.columns:
                    if 'min' in range_dict:
                        df = df[df[param] >= range_dict['min']]
                    if 'max' in range_dict:
                        df = df[df[param] <= range_dict['max']]
        
        # Calculate metrics
        total_sims = len(df)
        successful_sims = len(df[df['success'] == True])
        avg_roi = df['actual_roi'].mean()
        std_roi = df['actual_roi'].std()
        
        # Calculate Sharpe ratio (simplified)
        risk_free_rate = 0.02  # 2% annual
        excess_returns = df['actual_roi'] - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + df['actual_roi'] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Find best and worst market conditions
        best_idx = df['actual_roi'].idxmax() if not df.empty else None
        worst_idx = df['actual_roi'].idxmin() if not df.empty else None
        
        best_conditions = {}
        worst_conditions = {}
        
        if best_idx:
            best_row = df.loc[best_idx]
            best_conditions = {k.replace('market_', ''): v 
                             for k, v in best_row.items() 
                             if k.startswith('market_')}
        
        if worst_idx:
            worst_row = df.loc[worst_idx]
            worst_conditions = {k.replace('market_', ''): v 
                              for k, v in worst_row.items() 
                              if k.startswith('market_')}
        
        return BacktestResult(
            strategy_name=strategy.get('name', 'Unknown'),
            total_simulations=total_sims,
            successful_simulations=successful_sims,
            average_roi=avg_roi,
            std_roi=std_roi,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            best_market_conditions=best_conditions,
            worst_market_conditions=worst_conditions
        )
    
    def analyze_parameter_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """Analyze sensitivity of ROI to each parameter"""
        if self.df is None or self.df.empty:
            return {}
        
        sensitivity = {}
        
        # Parameters to analyze
        params = ['total_supply', 'airdrop_percent', 'launch_price', 
                 'opportunity_cost', 'volatility', 'gas_cost']
        
        for param in params:
            if param not in self.df.columns:
                continue
            
            # Calculate correlation with ROI
            correlation = self.df[param].corr(self.df['actual_roi'])
            
            # Calculate impact (change in ROI per unit change in parameter)
            # Using linear regression coefficient
            from sklearn.linear_model import LinearRegression
            X = self.df[param].values.reshape(-1, 1)
            y = self.df['actual_roi'].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) > 10:  # Need enough data points
                reg = LinearRegression()
                reg.fit(X_clean, y_clean)
                impact = reg.coef_[0]
            else:
                impact = 0
            
            # Calculate importance (R-squared)
            if len(X_clean) > 10:
                r_squared = reg.score(X_clean, y_clean)
            else:
                r_squared = 0
            
            sensitivity[param] = {
                'correlation': correlation,
                'impact': impact,
                'importance': r_squared
            }
        
        return sensitivity
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if self.df is None or self.df.empty:
            logger.warning("No historical data for report generation")
            return {}
        
        report = {
            'summary': {
                'total_simulations': len(self.df),
                'date_range': {
                    'start': str(self.df.index.min()),
                    'end': str(self.df.index.max())
                },
                'success_rate': (self.df['success'].sum() / len(self.df) * 100),
                'average_roi': self.df['actual_roi'].mean(),
                'std_roi': self.df['actual_roi'].std(),
                'best_roi': self.df['actual_roi'].max(),
                'worst_roi': self.df['actual_roi'].min()
            },
            'parameter_sensitivity': self.analyze_parameter_sensitivity(),
            'track_performance': self._analyze_track_performance(),
            'market_condition_impact': self._analyze_market_conditions(),
            'time_series_analysis': self._analyze_time_series()
        }
        
        # Save report if path specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved to {output_path}")
        
        return report
    
    def _analyze_track_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by track"""
        track_performance = {}
        
        # Find track columns
        track_roi_cols = [col for col in self.df.columns if col.endswith('_roi') and not col.startswith('market_')]
        
        for col in track_roi_cols:
            track_name = col.replace('_roi', '')
            
            # Filter rows where this track was used
            track_data = self.df[self.df[col].notna()]
            
            if len(track_data) > 0:
                track_performance[track_name] = {
                    'avg_roi': track_data[col].mean(),
                    'std_roi': track_data[col].std(),
                    'success_rate': (track_data[col] > 0).sum() / len(track_data) * 100,
                    'usage_count': len(track_data)
                }
        
        return track_performance
    
    def _analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze impact of market conditions"""
        market_impact = {}
        
        # Find market condition columns
        market_cols = [col for col in self.df.columns if col.startswith('market_')]
        
        for col in market_cols:
            condition_name = col.replace('market_', '')
            
            # Divide into high/low conditions
            median_val = self.df[col].median()
            high_condition = self.df[self.df[col] > median_val]
            low_condition = self.df[self.df[col] <= median_val]
            
            market_impact[condition_name] = {
                'high_condition_avg_roi': high_condition['actual_roi'].mean(),
                'low_condition_avg_roi': low_condition['actual_roi'].mean(),
                'impact_difference': high_condition['actual_roi'].mean() - low_condition['actual_roi'].mean(),
                'correlation_with_roi': self.df[col].corr(self.df['actual_roi'])
            }
        
        return market_impact
    
    def _analyze_time_series(self) -> Dict[str, Any]:
        """Analyze time series patterns"""
        # Resample to monthly data
        monthly = self.df.resample('M').agg({
            'actual_roi': ['mean', 'std', 'count'],
            'success': 'sum'
        })
        
        # Calculate trend
        if len(monthly) > 3:
            from scipy import stats
            x = np.arange(len(monthly))
            y = monthly[('actual_roi', 'mean')].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend = 'improving' if slope > 0 else 'declining'
        else:
            trend = 'insufficient_data'
            slope = 0
        
        return {
            'monthly_average_roi': monthly[('actual_roi', 'mean')].to_dict(),
            'monthly_volatility': monthly[('actual_roi', 'std')].to_dict(),
            'trend': trend,
            'trend_slope': slope
        }
    
    def predict_future_performance(self, strategy: Dict[str, Any], 
                                 market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Predict future performance based on historical data"""
        if self.df is None or self.df.empty:
            return {'predicted_roi': 0, 'confidence': 0}
        
        # Simple prediction using parameter sensitivity
        sensitivity = self.analyze_parameter_sensitivity()
        
        predicted_roi = self.df['actual_roi'].mean()  # Start with average
        
        # Adjust based on parameter differences
        for param, value in strategy.get('parameters', {}).items():
            if param in sensitivity and param in self.df.columns:
                avg_value = self.df[param].mean()
                impact = sensitivity[param]['impact']
                predicted_roi += impact * (value - avg_value)
        
        # Adjust based on market conditions
        market_impact = self._analyze_market_conditions()
        for condition, value in market_conditions.items():
            if condition in market_impact:
                # Simple linear adjustment
                correlation = market_impact[condition]['correlation_with_roi']
                predicted_roi += correlation * (value - 0.5) * 10  # Normalize around 0.5
        
        # Calculate confidence based on data availability
        relevant_data_points = len(self.df)
        confidence = min(relevant_data_points / 100, 1.0)  # Max confidence at 100 data points
        
        return {
            'predicted_roi': predicted_roi,
            'confidence': confidence,
            'data_points_used': relevant_data_points
        }
    
    def plot_historical_performance(self, save_path: Optional[str] = None):
        """Plot historical performance charts"""
        if self.df is None or self.df.empty:
            logger.warning("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROI over time
        ax = axes[0, 0]
        self.df['actual_roi'].plot(ax=ax, style='o-', alpha=0.7)
        ax.set_title('ROI Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('ROI (%)')
        ax.grid(True, alpha=0.3)
        
        # 2. Success rate by month
        ax = axes[0, 1]
        monthly_success = self.df.resample('M')['success'].agg(['sum', 'count'])
        monthly_success['rate'] = monthly_success['sum'] / monthly_success['count'] * 100
        monthly_success['rate'].plot(ax=ax, kind='bar')
        ax.set_title('Monthly Success Rate')
        ax.set_xlabel('Month')
        ax.set_ylabel('Success Rate (%)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Parameter sensitivity heatmap
        ax = axes[1, 0]
        sensitivity = self.analyze_parameter_sensitivity()
        if sensitivity:
            sens_df = pd.DataFrame(sensitivity).T
            sns.heatmap(sens_df[['correlation', 'impact']], 
                       annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax)
            ax.set_title('Parameter Sensitivity')
        
        # 4. Track performance comparison
        ax = axes[1, 1]
        track_perf = self._analyze_track_performance()
        if track_perf:
            track_df = pd.DataFrame(track_perf).T
            if 'avg_roi' in track_df.columns:
                track_df['avg_roi'].plot(ax=ax, kind='bar')
                ax.set_title('Average ROI by Track')
                ax.set_xlabel('Track')
                ax.set_ylabel('Average ROI (%)')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Historical performance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_mock_historical_data(num_simulations: int = 100, 
                              output_path: str = "data/historical_simulations.json"):
    """Create mock historical data for testing"""
    analyzer = HistoricalAnalyzer()
    
    # Generate mock simulations
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_simulations):
        # Random parameters within reasonable ranges
        params = {
            'total_supply': np.random.uniform(1e8, 1e10),
            'airdrop_percent': np.random.uniform(5, 25),
            'launch_price': np.random.uniform(0.05, 0.5),
            'opportunity_cost': np.random.uniform(5, 15),
            'volatility': np.random.uniform(40, 100),
            'gas_cost': np.random.uniform(20, 60)
        }
        
        # Random market conditions
        market_conditions = {
            'btc_price': np.random.uniform(20000, 60000),
            'eth_price': np.random.uniform(1000, 4000),
            'market_sentiment': np.random.uniform(0, 1),  # 0 = bearish, 1 = bullish
            'gas_price_gwei': np.random.uniform(10, 100)
        }
        
        # Calculate mock ROI (simplified)
        base_roi = -50 + (params['airdrop_percent'] * 2) + (params['launch_price'] * 100)
        market_adjustment = (market_conditions['market_sentiment'] - 0.5) * 50
        volatility_penalty = -params['volatility'] * 0.3
        
        actual_roi = base_roi + market_adjustment + volatility_penalty + np.random.normal(0, 20)
        
        # Create track results
        track_results = {
            'NODE_OPERATOR': {
                'roi': actual_roi + np.random.normal(0, 10),
                'allocation': np.random.uniform(1000, 10000)
            },
            'RISK_UNDERWRITER': {
                'roi': actual_roi + np.random.normal(-10, 15),
                'allocation': np.random.uniform(500, 5000)
            }
        }
        
        # Create simulation result
        result = SimulationResult(
            timestamp=(start_date + timedelta(days=i*3.65)).isoformat(),
            parameters=params,
            track_results=track_results,
            actual_roi=actual_roi,
            success=actual_roi > 0,
            market_conditions=market_conditions,
            notes=f"Mock simulation {i+1}"
        )
        
        analyzer.save_simulation(result, output_path)
    
    logger.info(f"Created {num_simulations} mock historical simulations")
    return output_path