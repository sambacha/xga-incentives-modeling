from datetime import datetime
import json
from typing import Optional

from .types import AirdropParameters
from .core import AirdropCalculator

def generate_recommendation_report(params: AirdropParameters, 
                                 calculator: Optional[AirdropCalculator] = None,
                                 save_path: str = "airdrop_recommendations.md") -> str:
    """
    Generate comprehensive recommendation report
    
    But wait, maybe it's better... to include more specific
    recommendations based on the analysis results
    """
    if calculator is None:
        calculator = AirdropCalculator(params)
    
    metrics = calculator.calculate_market_metrics()
    
    report = f"""# Airdrop Parameter Recommendations Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

Based on the exotic options framework analysis:

- **Minimum Market Cap Required**: ${metrics.min_market_cap/1e6:.1f}M
- **Expected Profitable Users**: {metrics.profitable_users_percent:.1f}%
- **Average User ROI**: {metrics.avg_roi:.1f}%
- **Optimal Capital Commitment**: ${metrics.optimal_capital:,.0f}
- **Hurdle Rate Multiple**: {metrics.hurdle_rate:.2f}x
- **Beta Value**: {metrics.beta_value:.3f}

## Risk Assessment

### Market Conditions
- **Opportunity Cost**: {params.opportunity_cost}% APY
- **Risk Level**: {"⚠️ HIGH" if params.opportunity_cost > 15 else "✅ MODERATE" if params.opportunity_cost > 8 else "✅ LOW"}
- **Recommendation**: {"Consider delaying launch" if params.opportunity_cost > 15 else "Good market conditions"}

### User Segment Analysis
"""
    
    for segment in metrics.segment_results:
        emoji = "✅" if segment.profitable else "❌"
        report += f"\n{emoji} **{segment.segment}**"
        report += f"\n   - ROI: {segment.roi:.1f}%"
        report += f"\n   - Min tokens needed: {segment.min_tokens:,.0f}"
        report += f"\n   - Expected allocation: {segment.estimated_allocation:,.0f}"
        report += f"\n   - Status: {'Profitable' if segment.profitable else 'Unprofitable'}\n"
    
    # Strategic recommendations
    report += f"""
## Strategic Recommendations

### 1. Timing Strategy
"""
    if params.opportunity_cost > 15:
        report += "- **Delay Launch**: High opportunity costs will exclude most users\n"
        report += "- Wait for market yields to compress below 10%\n"
        report += "- Consider shorter campaign duration to reduce opportunity cost impact\n"
    elif params.opportunity_cost > 10:
        report += "- **Proceed with Caution**: Moderate opportunity costs\n"
        report += "- Focus on whale and institutional segments\n"
        report += "- Consider increasing airdrop allocation to improve profitability\n"
    else:
        report += "- **Optimal Launch Window**: Low opportunity costs favor participation\n"
        report += "- Broad user participation expected\n"
        report += "- Consider extending campaign to maximize participation\n"
    
    report += f"""
### 2. Token Allocation
- Current allocation of {params.airdrop_percent}% is {"appropriate" if 15 <= params.airdrop_percent <= 35 else "consider adjusting to 20-30% range"}
- Total airdrop value: ${params.total_supply * params.airdrop_percent / 100 * params.launch_price / 1e6:.1f}M

### 3. Market Cap Target
- Minimum viable market cap: ${metrics.min_market_cap/1e6:.0f}M
- Recommended target: ${metrics.min_market_cap * 1.5 / 1e6:.0f}M (50% buffer)
- Critical threshold: ${metrics.min_market_cap * 0.8 / 1e6:.0f}M (below this, <30% profitable)

### 4. Gas Optimization
"""
    if params.gas_cost > 100:
        report += "- **Critical**: High gas costs exclude retail users\n"
        report += "- Deploy on L2 (Arbitrum, Optimism, Polygon)\n"
        report += "- Implement gas rebate program\n"
        report += "- Consider batch claiming mechanisms\n"
    elif params.gas_cost > 50:
        report += "- **Important**: Moderate gas costs impact smaller users\n"
        report += "- Consider L2 deployment for better accessibility\n"
        report += "- Implement efficient claim contracts\n"
    else:
        report += "- Gas costs are reasonable\n"
        report += "- Current infrastructure is acceptable\n"
    
    report += f"""
### 5. Vesting Strategy
- Current: {params.immediate_unlock}% immediate, {100 - params.immediate_unlock}% over {params.vesting_months} months
- {"Consider shorter vesting" if params.vesting_months > 18 else "Vesting period is appropriate"}
- Implement cliff periods to prevent immediate dumping

## Implementation Checklist

- [ ] Secure audit of distribution contracts
- [ ] Implement Merkle tree for gas-efficient claims
- [ ] Set up multi-sig for distribution control
- [ ] Create claiming interface with clear instructions
- [ ] Implement sybil detection mechanisms
- [ ] Prepare liquidity for launch
- [ ] Set up monitoring dashboards
- [ ] Create comprehensive documentation

## Risk Mitigation

1. **Sybil Attacks**: Implement Gitcoin Passport, on-chain analysis, time-based requirements
2. **Market Manipulation**: Vesting schedules, liquidity requirements, anti-dump mechanisms
3. **Regulatory**: Clear utility documentation, geographic restrictions where needed
4. **Technical**: Comprehensive testing, bug bounties, gradual rollout

## Conclusion

{"The current parameters are well-optimized for broad participation." if metrics.profitable_users_percent > 60 else "Consider adjusting parameters to improve user profitability."}
Market conditions {"are favorable" if params.opportunity_cost < 10 else "require careful consideration"} for launch.

---
*This report is for informational purposes only and does not constitute financial advice.*
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    return report
