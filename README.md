
## Core Constraints
- Price range: $0.10 ≤ p ≤ $0.50
- Supply: 1B ≤ S ≤ 50B tokens
- Market cap: MC = p × S
- Airdrop percentage: 30% of total supply

## Allocation Models

### Linear Model
```
allocation_ratio = capital / 1,000,000
allocation = total_airdrop × allocation_ratio × 0.01
```

### Quadratic Model
```
sqrt_capital = √(min(capital, 1,000,000))
allocation_ratio = sqrt_capital / 10,000
allocation = total_airdrop × allocation_ratio × 0.1
```

### Logarithmic Model
```
log_capital = log₁₀(max(capital, 100))
allocation_ratio = log_capital / 6
allocation = total_airdrop × allocation_ratio × 0.05
```

### Tiered Model
```
multiplier = {
    1.5  if capital < 1,000
    1.2  if 1,000 ≤ capital < 10,000
    1.0  if 10,000 ≤ capital < 100,000
    0.8  if capital ≥ 100,000
}
allocation = base_allocation × multiplier
```

## Track-Specific Calculations

### Node Operator
- Capital: `points × $5,000 ÷ validators_operated`
- Points: `validators × duration × performance × uptime`
- Risk factor: 0.8
- Model: Quadratic

### Risk Underwriter
- Capital: `staking_amount × token_price`
- Duration multiplier: `1.0 (≤6mo) | 1.5 (6-12mo) | 1.75 (12-24mo) | 2.0 (>24mo)`
- Risk factor: 1.2
- Model: Linear

### Liquidity Provider
- Capital: `LST_amount × $3,000`
- Points: `LST_amount × (duration/12) × bonus_multiplier`
- Risk factor: 0.6 (stable) | 1.0 (standard)
- Model: Quadratic

### Auction Participant
- Capital: `performance_adjusted_bid_value`
- Performance: `success_rate × bid_accuracy`
- Risk factor: 0.8 - 1.5 (variable)
- Model: Tiered

## Multi-Track Aggregation

### Weighted Risk Factor
```
risk_weighted = Σ(capital_i × risk_factor_i) / Σ(capital_i)
```

### Combined Allocation
```
dominant_model = argmax(capital_per_track)
allocation_ratio = capital_equivalent / total_capital_equivalent
final_allocation = total_airdrop × allocation_ratio × model_multiplier
```

## Options Pricing Framework

### Beta Calculation
```
β = 1 + (r/σ²) + √((r/σ²)² + 2r/σ²)
Bounded: 1.2 ≤ β ≤ 2.0
```

### Hurdle Rate
```
H = β/(β-1)
Bounded: 1.2 ≤ H ≤ 3.0
```

### Required Return
```
R = H - 1
Range: 20% - 200%
```

## Economic Parameters

| Parameter | Value |
|-----------|-------|
| Opportunity cost (r) | 0.5% - 1.0% |
| Volatility (σ) | 20% |
| Campaign duration | 3+ months |
| Revenue share | 40% |
| Airdrop percentage | 25% - 30% |

## Constraint Solving

### Objective Function
```
minimize: w₁(|MC - MC_target|) + w₂(|p - p_target|) + w₃(1/user_profit)
```

### Hard Constraints
```
0.10 ≤ price ≤ 0.50
supply ≥ 1,000,000,000
airdrop_value ≥ 0
user_profit ≥ 1.0
```

### Soft Constraints (Weighted)
```
market_cap ≈ target_mc × relaxation_factor
relaxation_factor ∈ {1.0, 0.8, 0.6, 0.4, 0.2}
```

## Solution Validation
```
valid = (price ∈ [0.10, 0.50]) ∧
        (supply ≥ 1B) ∧
        (airdrop_value > 0) ∧
        (user_profit ≥ 1.0) ∧
        (hurdle_rate ∈ [1.2, 3.0])
```
