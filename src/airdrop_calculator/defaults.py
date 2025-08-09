"""
Optimized default parameters for realistic ROI projections
"""

# Optimized default parameters based on analysis
OPTIMIZED_DEFAULTS = {
    'total_supply': 1_000_000_000,
    'airdrop_percent': 15.0,  # Increased from 10.0
    'launch_price': 0.35,  # Increased from 0.10
    'opportunity_cost': 8.0,  # Reduced from 10.0
    'volatility': 60.0,  # Reduced from 80.0
    'gas_cost': 30.0,  # Reduced from 50.0
    'campaign_duration': 12,
    'airdrop_certainty': 95.0,  # Increased from 90.0
    'revenue_share': 0.0,
    'vesting_months': 18,
    'immediate_unlock': 30.0
}

# Track-specific capital requirements (adjusted for better ROI)
TRACK_CAPITAL_REQUIREMENTS = {
    'NODE_OPERATOR': {
        'base_capital_per_validator': 5000,  # Reduced from 10000
        'min_validators': 1,
        'max_validators': 100
    },
    'RISK_UNDERWRITER': {
        'token_prices': {
            'FOLD': 1.0,
            'EIGEN': 3.0
        },
        'min_stake': 1000,
        'max_stake': 1000000
    },
    'LIQUIDITY_PROVIDER': {
        'lst_price': 3000,  # Updated ETH price
        'min_lst_amount': 0.1,
        'max_lst_amount': 1000
    },
    'AUCTION_PARTICIPANT': {
        'min_bid_value': 100,
        'max_bid_value': 10000000,
        'performance_multiplier': 1.5  # Increased from 1.0
    }
}

# Realistic hurdle rate bounds
HURDLE_RATE_BOUNDS = {
    'min': 1.2,  # 20% minimum required return
    'max': 3.0,  # 200% maximum required return
    'default': 1.5  # 50% default required return
}