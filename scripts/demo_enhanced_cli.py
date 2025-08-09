#!/usr/bin/env python3
"""
Demo script showing the enhanced CLI with comprehensive calculation results
"""

import subprocess
import json
from pathlib import Path

def run_cli_command(command):
    """Run a CLI command and capture output"""
    print("="*80)
    print(f"RUNNING: {command}")
    print("="*80)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result

def demo_enhanced_cli():
    """Demonstrate the enhanced CLI features"""
    
    print("🚀 ENHANCED AIRDROP CALCULATOR CLI DEMO")
    print("="*80)
    
    # 1. Demo comprehensive calculations
    print("\n1️⃣ COMPREHENSIVE CALCULATIONS COMMAND")
    run_cli_command(
        "python -m airdrop_calculator.cli calculate "
        "--supply 1000000000 --airdrop-percent 25 --price 0.75 "
        "--opportunity-cost 8 --volatility 60 --gas-cost 30 "
        "--revenue-share 15"
    )
    
    # 2. Demo enhanced solve command
    print("\n2️⃣ ENHANCED SOLVE COMMAND WITH DETAILED OUTPUT")
    run_cli_command(
        "python -m airdrop_calculator.cli solve "
        "--market-cap 300000000 --profitable-users 65 --method soft"
    )
    
    # 3. Demo analyze command with comprehensive output
    print("\n3️⃣ ENHANCED ANALYZE COMMAND")
    run_cli_command(
        "python -m airdrop_calculator.cli analyze "
        "--supply 500000000 --airdrop-percent 30 --price 1.20 "
        "--opportunity-cost 12 --volatility 70"
    )
    
    # 4. Create a sample config and demo config-based calculation
    print("\n4️⃣ CONFIG-BASED COMPREHENSIVE CALCULATION")
    
    sample_config = {
        "total_supply": 2000000000,
        "airdrop_percent": 18,
        "launch_price": 0.25,
        "opportunity_cost": 6,
        "volatility": 90,
        "gas_cost": 75,
        "campaign_duration": 9,
        "airdrop_certainty": 85,
        "revenue_share": 20,
        "vesting_months": 24,
        "immediate_unlock": 25
    }
    
    config_file = Path("demo_config.json")
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    run_cli_command(
        f"python -m airdrop_calculator.cli calculate --config {config_file}"
    )
    
    # 5. Demo profitability analysis
    print("\n5️⃣ PROFITABILITY ANALYSIS")
    run_cli_command(
        "python -m airdrop_calculator.cli analyze-profitability "
        "--capital 10000 --opportunity-cost 0.15 --expected-share 0.001 "
        "--total-airdrop 50000000 --time-months 8"
    )
    
    # Cleanup
    if config_file.exists():
        config_file.unlink()
    
    print("\n🎉 DEMO COMPLETED!")
    print("="*80)
    print("ENHANCED CLI FEATURES DEMONSTRATED:")
    print("✅ Comprehensive calculation breakdowns")
    print("✅ Step-by-step option pricing formulas")
    print("✅ Detailed investment analysis")
    print("✅ Market context and categorization")
    print("✅ Profitability thresholds for different timeframes")
    print("✅ Token distribution and vesting schedules")
    print("✅ Allocation model comparisons")
    print("✅ Sensitivity analysis")
    print("✅ User segment analysis")
    print("✅ Cliff value calculations")
    print("="*80)

if __name__ == "__main__":
    demo_enhanced_cli()