# Chapter 309: Reward Shaping for Trading

## Introduction

Reinforcement learning (RL) has emerged as a powerful paradigm for building autonomous trading agents. However, one of the most critical and often overlooked challenges in applying RL to financial markets is the design of the reward function. A naive reward signal — such as raw profit and loss (PnL) — can lead to agents that overfit to specific market conditions, take excessive risks, or fail to learn meaningful policies due to sparse and delayed feedback.

**Reward shaping** addresses this problem by augmenting the base reward signal with additional terms that guide the agent toward desirable behaviors without changing the optimal policy. The key insight from the seminal work of Ng, Harada, and Russell (1999) is that potential-based reward shaping preserves policy invariance: the optimal policy under the shaped reward is identical to the optimal policy under the original reward, but learning can be dramatically faster.

In the context of trading, reward shaping allows us to encode domain knowledge — such as preferences for risk-adjusted returns, aversion to drawdowns, and sensitivity to transaction costs — directly into the learning signal. This chapter explores the theory behind reward shaping, presents several practical reward designs for trading, and provides a complete Rust implementation with integration to Bybit exchange data.

## Mathematical Foundations

### The Reward Shaping Framework

Consider a Markov Decision Process (MDP) defined by the tuple (S, A, T, R, gamma), where S is the state space, A is the action space, T is the transition function, R is the reward function, and gamma is the discount factor.

A shaped reward function R' is defined as:

```
R'(s, a, s') = R(s, a, s') + F(s, a, s')
```

where F is the shaping function that provides additional feedback to the agent.

### Potential-Based Shaping

The critical theoretical result is that **potential-based shaping** guarantees policy invariance. The shaping function takes the form:

```
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

where Phi: S -> R is a real-valued potential function defined over states. This formulation ensures that:

1. **Policy Invariance Theorem**: For any MDP M and any potential function Phi, the optimal policy under the shaped reward R' = R + F is identical to the optimal policy under the original reward R. This holds because the potential terms telescope across trajectories, contributing only boundary terms that do not affect the relative ordering of policies.

2. **Convergence Guarantee**: Q-learning and SARSA with potential-based shaping converge to the same optimal Q-values (up to a constant offset) as without shaping. Specifically, Q'*(s, a) = Q*(s, a) - Phi(s), where Q'* and Q* are the optimal Q-values under shaped and unshaped rewards respectively.

3. **Faster Learning**: While the optimal policy is preserved, the shaped reward can dramatically reduce the number of episodes needed for convergence by providing denser feedback and reducing the variance of reward estimates.

### Proof Sketch of Policy Invariance

For any trajectory tau = (s_0, a_0, s_1, a_1, ..., s_T), the cumulative shaped reward differs from the cumulative original reward by:

```
Sum_{t=0}^{T-1} F(s_t, a_t, s_{t+1}) = Sum_{t=0}^{T-1} [gamma * Phi(s_{t+1}) - Phi(s_t)]
```

This telescoping sum equals:

```
gamma^T * Phi(s_T) - Phi(s_0) + Sum_{t=1}^{T-1} (gamma - 1) * Phi(s_t) * [correction terms]
```

In the discounted infinite-horizon case, this reduces to a constant offset that depends only on the initial state, not on the policy. Therefore, argmax over policies is preserved.

### Non-Potential-Based Shaping and Risks

If the shaping function F is not derived from a potential, policy invariance is **not** guaranteed. Common pitfalls include:

- **Positive reward cycles**: The agent may find loops in state space that accumulate shaping reward without making progress.
- **Policy distortion**: The optimal policy under shaped rewards may differ from the true optimal policy.
- **Convergence to suboptimal equilibria**: The agent may settle on behaviors that exploit the shaping reward rather than optimizing the true objective.

In practice, many useful trading reward modifications (such as drawdown penalties) are not strictly potential-based. We address this by analyzing the trade-offs and providing guidelines for when non-potential-based shaping is acceptable.

## Reward Designs for Trading

### 1. Base Reward: Raw PnL

The simplest reward is the change in portfolio value:

```
R_pnl(t) = V(t) - V(t-1)
```

where V(t) is the portfolio value at time t. While intuitive, this reward:
- Provides no risk adjustment
- Can lead to high-variance policies
- Does not penalize drawdowns or excessive trading

### 2. Sharpe Ratio-Based Shaping

The Sharpe ratio measures risk-adjusted returns. We can incorporate it through a potential function:

```
Phi_sharpe(s) = k * rolling_sharpe(s, window)
```

where `rolling_sharpe` computes the Sharpe ratio over a recent window of returns, and k is a scaling constant. The shaping reward becomes:

```
F_sharpe(s, s') = gamma * k * sharpe(s') - k * sharpe(s)
```

This encourages the agent to transition to states with higher risk-adjusted returns. When the rolling Sharpe ratio improves, the agent receives a positive shaping bonus; when it deteriorates, the agent is penalized.

**Design considerations:**
- Window size affects sensitivity: shorter windows (20-50 steps) are more responsive but noisier
- The scaling factor k should be calibrated relative to the magnitude of PnL rewards
- Annualization factor: multiply by sqrt(252) for daily data, sqrt(252 * 24 * 60) for minute data

### 3. Drawdown Penalty Shaping

Maximum drawdown is a critical risk metric for traders. We define a drawdown penalty potential:

```
Phi_dd(s) = -lambda * max(0, peak_value(s) - current_value(s)) / peak_value(s)
```

where lambda controls the severity of the penalty and `peak_value(s)` is the historical peak portfolio value. The resulting shaping signal penalizes the agent when drawdown increases and rewards recovery from drawdowns.

**Properties:**
- The penalty grows as the portfolio moves further from its peak
- Recovery toward the peak generates positive shaping reward
- The parameter lambda allows fine-tuning the risk-return trade-off
- Note: this is approximately potential-based when the peak value changes slowly

### 4. Transaction Cost Penalty

Excessive trading erodes returns through commissions and slippage. A transaction cost shaping term:

```
F_tc(s, a, s') = -c * |position_change(a)|
```

where c is the cost per unit traded. This is not potential-based (it depends on the action), but it directly models the real-world cost structure and is widely used in practice.

**Extensions:**
- Proportional costs: `c * |position_change| * price`
- Tiered fee structures: different rates for maker vs taker orders
- Slippage modeling: costs that increase with order size relative to market depth

### 5. Regime-Aware Shaping

Market regimes (trending, mean-reverting, volatile) affect optimal trading strategies. A regime-aware potential function:

```
Phi_regime(s) = w_regime(s) * base_potential(s)
```

where `w_regime(s)` is a regime-dependent weight. For example:
- In trending regimes: increase weight on momentum-following rewards
- In mean-reverting regimes: increase weight on contrarian rewards
- In high-volatility regimes: increase weight on risk penalties

Regime detection can be based on rolling volatility, moving average crossovers, or hidden Markov models.

### 6. Composite Reward Function

In practice, we combine multiple shaping terms:

```
R_total(s, a, s') = R_pnl(s, a, s') + alpha * F_sharpe(s, s') + beta * F_dd(s, s') + delta * F_tc(s, a, s')
```

where alpha, beta, and delta are hyperparameters controlling the relative importance of each component. These can be tuned via:
- Grid search over validation episodes
- Multi-objective optimization (Pareto frontier analysis)
- Adaptive weighting based on training progress

## Rust Implementation

The implementation in `rust/src/lib.rs` provides:

1. **`RewardShaper` trait**: A common interface for all reward shaping strategies, with methods for computing shaped rewards and updating internal state.

2. **`PnlReward`**: Base PnL reward computation from price changes and positions.

3. **`SharpeShapingReward`**: Potential-based shaping using rolling Sharpe ratio, with configurable window size and scaling factor.

4. **`DrawdownPenaltyReward`**: Drawdown-based shaping that tracks peak portfolio value and penalizes deviations.

5. **`TransactionCostPenalty`**: Action-dependent cost modeling with configurable fee rates.

6. **`CompositeReward`**: Combines multiple reward components with weighted aggregation.

7. **`QLearningAgent`**: A tabular Q-learning agent that supports shaped rewards, with epsilon-greedy exploration and configurable learning parameters.

8. **`BybitClient`**: HTTP client for fetching historical kline data from the Bybit API.

The implementation emphasizes:
- Type safety through Rust's strong type system
- Zero-cost abstractions for reward composition
- Efficient state discretization for Q-learning
- Comprehensive unit testing

## Bybit Data Integration

The system fetches historical OHLCV data from Bybit's public API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The data pipeline:
1. Fetch kline data with configurable symbol, interval, and limit
2. Parse JSON response into typed Rust structures
3. Compute derived features (returns, volatility, moving averages)
4. Feed into the trading simulation environment
5. Run Q-learning episodes with different reward shaping configurations

## Experimental Comparison

The trading example (`rust/examples/trading_example.rs`) demonstrates the impact of reward shaping by training three agents:

1. **Raw PnL Agent**: Uses only price-change reward. Tends to take aggressive positions, high variance in performance.

2. **Sharpe-Shaped Agent**: Adds rolling Sharpe ratio potential. Learns smoother equity curves, better risk-adjusted returns.

3. **Drawdown-Penalized Agent**: Adds drawdown penalty. More conservative, smaller maximum drawdown, potentially lower total return but better Sortino ratio.

The comparison metrics include:
- Total return
- Sharpe ratio
- Maximum drawdown
- Sortino ratio
- Number of trades (turnover)
- Win rate

## Key Takeaways

1. **Reward shaping is essential** for practical RL trading systems. Raw PnL alone provides insufficient guidance for learning risk-aware policies.

2. **Potential-based shaping preserves optimality**. The policy invariance theorem guarantees that potential-based shaping functions do not alter the optimal policy while accelerating learning.

3. **Domain knowledge matters**. Effective reward shaping encodes financial intuition — Sharpe ratios, drawdown aversion, transaction costs — into the learning signal.

4. **Composite rewards outperform individual components**. Combining multiple shaping terms with carefully tuned weights produces the most robust trading agents.

5. **Non-potential-based shaping requires caution**. Transaction cost penalties and some risk adjustments are not strictly potential-based. Monitor for policy distortion and validate against unshaped baselines.

6. **Hyperparameter sensitivity**. The weights on shaping terms (alpha, beta, delta) significantly affect learned behavior. Use systematic tuning with out-of-sample validation.

7. **Regime awareness improves adaptability**. Adjusting reward shaping based on detected market regimes helps agents adapt to changing market conditions.

8. **Rust provides performance advantages**. The computational demands of RL training — many episodes, many time steps — benefit from Rust's zero-cost abstractions and efficient memory management, enabling faster experimentation cycles.

## References

- Ng, A. Y., Harada, D., & Russell, S. J. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.
- Devlin, S., & Kudenko, D. (2012). Dynamic potential-based reward shaping. *AAMAS*.
- Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*.
- Almahdi, S., & Yang, S. Y. (2017). An adaptive portfolio trading system: A risk-return portfolio optimization using recurrent reinforcement learning with expected maximum drawdown. *Expert Systems with Applications*.
- Spooner, T., Fearnley, J., Savani, R., & Kouroupi, A. (2018). Market making via reinforcement learning. *AAMAS*.
