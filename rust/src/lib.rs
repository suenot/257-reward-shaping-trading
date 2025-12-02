use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::Rng;

// ============================================================================
// Data Structures
// ============================================================================

/// A single OHLCV candle from market data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Trading action: Buy, Sell, or Hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Buy,
    Sell,
    Hold,
}

impl Action {
    pub fn all() -> &'static [Action] {
        &[Action::Buy, Action::Sell, Action::Hold]
    }

    pub fn index(&self) -> usize {
        match self {
            Action::Buy => 0,
            Action::Sell => 1,
            Action::Hold => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Action::Buy,
            1 => Action::Sell,
            _ => Action::Hold,
        }
    }
}

/// Trading state for the environment.
#[derive(Debug, Clone)]
pub struct TradingState {
    pub step: usize,
    pub position: f64,       // current position: -1, 0, or 1
    pub portfolio_value: f64,
    pub peak_value: f64,
    pub returns_history: Vec<f64>,
    pub price: f64,
    pub prev_price: f64,
}

impl TradingState {
    pub fn new(initial_value: f64, price: f64) -> Self {
        Self {
            step: 0,
            position: 0.0,
            portfolio_value: initial_value,
            peak_value: initial_value,
            returns_history: Vec::new(),
            price,
            prev_price: price,
        }
    }

    /// Discretize state for tabular Q-learning.
    pub fn discretize(&self) -> u64 {
        let pos_bin = if self.position < -0.5 {
            0u64
        } else if self.position > 0.5 {
            2
        } else {
            1
        };

        let ret = if self.prev_price > 0.0 {
            (self.price - self.prev_price) / self.prev_price
        } else {
            0.0
        };
        let ret_bin = if ret < -0.02 {
            0u64
        } else if ret < -0.005 {
            1
        } else if ret < 0.005 {
            2
        } else if ret < 0.02 {
            3
        } else {
            4
        };

        let dd = if self.peak_value > 0.0 {
            (self.peak_value - self.portfolio_value) / self.peak_value
        } else {
            0.0
        };
        let dd_bin = if dd < 0.01 {
            0u64
        } else if dd < 0.05 {
            1
        } else {
            2
        };

        pos_bin * 15 + ret_bin * 3 + dd_bin
    }
}

// ============================================================================
// Reward Shaping Trait and Implementations
// ============================================================================

/// Trait for reward shaping strategies.
pub trait RewardShaper {
    /// Compute the shaped reward given previous and current states and the action taken.
    fn shape(
        &mut self,
        prev_state: &TradingState,
        action: Action,
        curr_state: &TradingState,
    ) -> f64;

    /// Reset internal state for a new episode.
    fn reset(&mut self);

    /// Name of this shaping strategy.
    fn name(&self) -> &str;
}

/// Base PnL reward: change in portfolio value.
#[derive(Debug, Clone)]
pub struct PnlReward;

impl PnlReward {
    pub fn new() -> Self {
        Self
    }

    /// Compute raw PnL from position and price change.
    pub fn compute(position: f64, price: f64, prev_price: f64) -> f64 {
        if prev_price == 0.0 {
            return 0.0;
        }
        position * (price - prev_price) / prev_price
    }
}

impl Default for PnlReward {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardShaper for PnlReward {
    fn shape(
        &mut self,
        prev_state: &TradingState,
        _action: Action,
        curr_state: &TradingState,
    ) -> f64 {
        Self::compute(prev_state.position, curr_state.price, prev_state.price)
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "PnL"
    }
}

/// Sharpe ratio-based potential shaping.
#[derive(Debug, Clone)]
pub struct SharpeShapingReward {
    pub window: usize,
    pub scale: f64,
    pub gamma: f64,
}

impl SharpeShapingReward {
    pub fn new(window: usize, scale: f64, gamma: f64) -> Self {
        Self {
            window,
            scale,
            gamma,
        }
    }

    /// Compute rolling Sharpe ratio from a returns history.
    pub fn rolling_sharpe(returns: &[f64], window: usize) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        let start = if returns.len() > window {
            returns.len() - window
        } else {
            0
        };
        let slice = &returns[start..];
        if slice.len() < 2 {
            return 0.0;
        }

        let n = slice.len() as f64;
        let mean = slice.iter().sum::<f64>() / n;
        let variance = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            0.0
        } else {
            mean / std_dev
        }
    }

    fn potential(&self, state: &TradingState) -> f64 {
        self.scale * Self::rolling_sharpe(&state.returns_history, self.window)
    }
}

impl RewardShaper for SharpeShapingReward {
    fn shape(
        &mut self,
        prev_state: &TradingState,
        _action: Action,
        curr_state: &TradingState,
    ) -> f64 {
        let prev_potential = self.potential(prev_state);
        let curr_potential = self.potential(curr_state);
        self.gamma * curr_potential - prev_potential
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "Sharpe Shaping"
    }
}

/// Drawdown penalty shaping.
#[derive(Debug, Clone)]
pub struct DrawdownPenaltyReward {
    pub lambda: f64,
    pub gamma: f64,
}

impl DrawdownPenaltyReward {
    pub fn new(lambda: f64, gamma: f64) -> Self {
        Self { lambda, gamma }
    }

    fn potential(&self, state: &TradingState) -> f64 {
        if state.peak_value <= 0.0 {
            return 0.0;
        }
        let drawdown = (state.peak_value - state.portfolio_value) / state.peak_value;
        -self.lambda * drawdown.max(0.0)
    }
}

impl RewardShaper for DrawdownPenaltyReward {
    fn shape(
        &mut self,
        prev_state: &TradingState,
        _action: Action,
        curr_state: &TradingState,
    ) -> f64 {
        let prev_potential = self.potential(prev_state);
        let curr_potential = self.potential(curr_state);
        self.gamma * curr_potential - prev_potential
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "Drawdown Penalty"
    }
}

/// Transaction cost penalty (action-dependent, not potential-based).
#[derive(Debug, Clone)]
pub struct TransactionCostPenalty {
    pub cost_rate: f64,
}

impl TransactionCostPenalty {
    pub fn new(cost_rate: f64) -> Self {
        Self { cost_rate }
    }

    /// Compute position change for an action given current position.
    pub fn position_change(action: Action, current_position: f64) -> f64 {
        let target = match action {
            Action::Buy => 1.0,
            Action::Sell => -1.0,
            Action::Hold => current_position,
        };
        (target - current_position).abs()
    }
}

impl RewardShaper for TransactionCostPenalty {
    fn shape(
        &mut self,
        prev_state: &TradingState,
        action: Action,
        _curr_state: &TradingState,
    ) -> f64 {
        let change = Self::position_change(action, prev_state.position);
        -self.cost_rate * change
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "Transaction Cost"
    }
}

/// Composite reward combining multiple shaping components.
pub struct CompositeReward {
    pub components: Vec<(f64, Box<dyn RewardShaper>)>,
    pub name: String,
}

impl CompositeReward {
    pub fn new(name: &str) -> Self {
        Self {
            components: Vec::new(),
            name: name.to_string(),
        }
    }

    pub fn add_component(&mut self, weight: f64, shaper: Box<dyn RewardShaper>) {
        self.components.push((weight, shaper));
    }
}

impl RewardShaper for CompositeReward {
    fn shape(
        &mut self,
        prev_state: &TradingState,
        action: Action,
        curr_state: &TradingState,
    ) -> f64 {
        self.components
            .iter_mut()
            .map(|(w, s)| *w * s.shape(prev_state, action, curr_state))
            .sum()
    }

    fn reset(&mut self) {
        for (_, s) in &mut self.components {
            s.reset();
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Potential-Based Shaping Wrapper
// ============================================================================

/// Generic potential-based shaping wrapper.
/// Takes any potential function Phi(s) and produces F = gamma * Phi(s') - Phi(s).
pub struct PotentialBasedShaping<F: Fn(&TradingState) -> f64> {
    pub potential_fn: F,
    pub gamma: f64,
    pub label: String,
}

impl<F: Fn(&TradingState) -> f64> PotentialBasedShaping<F> {
    pub fn new(potential_fn: F, gamma: f64, label: &str) -> Self {
        Self {
            potential_fn,
            gamma,
            label: label.to_string(),
        }
    }
}

impl<F: Fn(&TradingState) -> f64> RewardShaper for PotentialBasedShaping<F> {
    fn shape(
        &mut self,
        prev_state: &TradingState,
        _action: Action,
        curr_state: &TradingState,
    ) -> f64 {
        let prev_phi = (self.potential_fn)(prev_state);
        let curr_phi = (self.potential_fn)(curr_state);
        self.gamma * curr_phi - prev_phi
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        &self.label
    }
}

// ============================================================================
// Q-Learning Agent
// ============================================================================

/// Tabular Q-learning agent with shaped rewards.
pub struct QLearningAgent {
    pub q_table: HashMap<(u64, usize), f64>,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub min_epsilon: f64,
}

impl QLearningAgent {
    pub fn new(
        learning_rate: f64,
        discount_factor: f64,
        epsilon: f64,
        epsilon_decay: f64,
        min_epsilon: f64,
    ) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
            epsilon_decay,
            min_epsilon,
        }
    }

    pub fn get_q(&self, state: u64, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    pub fn best_action(&self, state: u64) -> Action {
        let mut best_a = 0;
        let mut best_q = f64::NEG_INFINITY;
        for a in 0..3 {
            let q = self.get_q(state, a);
            if q > best_q {
                best_q = q;
                best_a = a;
            }
        }
        Action::from_index(best_a)
    }

    pub fn select_action(&self, state: u64) -> Action {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            Action::from_index(rng.gen_range(0..3))
        } else {
            self.best_action(state)
        }
    }

    pub fn update(&mut self, state: u64, action: usize, reward: f64, next_state: u64) {
        let max_next_q = (0..3)
            .map(|a| self.get_q(next_state, a))
            .fold(f64::NEG_INFINITY, f64::max);

        let current_q = self.get_q(state, action);
        let new_q =
            current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);
        self.q_table.insert((state, action), new_q);
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }
}

// ============================================================================
// Trading Environment
// ============================================================================

/// Simple trading environment that steps through candle data.
pub struct TradingEnvironment {
    pub candles: Vec<Candle>,
    pub initial_balance: f64,
    pub state: TradingState,
    pub current_step: usize,
}

impl TradingEnvironment {
    pub fn new(candles: Vec<Candle>, initial_balance: f64) -> Self {
        let price = if candles.is_empty() {
            0.0
        } else {
            candles[0].close
        };
        Self {
            candles,
            initial_balance,
            state: TradingState::new(initial_balance, price),
            current_step: 0,
        }
    }

    pub fn reset(&mut self) -> TradingState {
        let price = if self.candles.is_empty() {
            0.0
        } else {
            self.candles[0].close
        };
        self.state = TradingState::new(self.initial_balance, price);
        self.current_step = 0;
        self.state.clone()
    }

    pub fn step(&mut self, action: Action) -> (TradingState, f64, bool) {
        self.current_step += 1;
        let done = self.current_step >= self.candles.len();
        if done {
            return (self.state.clone(), 0.0, true);
        }

        let prev_price = self.state.price;
        let new_price = self.candles[self.current_step].close;

        // Update position
        let new_position = match action {
            Action::Buy => 1.0,
            Action::Sell => -1.0,
            Action::Hold => self.state.position,
        };

        // Compute PnL
        let pnl = if prev_price > 0.0 {
            self.state.position * (new_price - prev_price) / prev_price * self.state.portfolio_value
        } else {
            0.0
        };

        let new_portfolio_value = self.state.portfolio_value + pnl;
        let ret = if self.state.portfolio_value > 0.0 {
            pnl / self.state.portfolio_value
        } else {
            0.0
        };

        let mut new_returns = self.state.returns_history.clone();
        new_returns.push(ret);

        let new_peak = new_portfolio_value.max(self.state.peak_value);

        self.state = TradingState {
            step: self.current_step,
            position: new_position,
            portfolio_value: new_portfolio_value,
            peak_value: new_peak,
            returns_history: new_returns,
            price: new_price,
            prev_price: prev_price,
        };

        let base_reward = PnlReward::compute(self.state.position, new_price, prev_price);
        (self.state.clone(), base_reward, false)
    }

    pub fn len(&self) -> usize {
        self.candles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }
}

// ============================================================================
// Training Utilities
// ============================================================================

/// Performance metrics for evaluation.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub sortino_ratio: f64,
    pub num_trades: usize,
    pub win_rate: f64,
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Return: {:.4}% | Sharpe: {:.4} | MaxDD: {:.4}% | Sortino: {:.4} | Trades: {} | WinRate: {:.2}%",
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.sortino_ratio,
            self.num_trades,
            self.win_rate * 100.0,
        )
    }
}

/// Compute performance metrics from a sequence of returns.
pub fn compute_metrics(returns: &[f64], num_trades: usize) -> PerformanceMetrics {
    let n = returns.len() as f64;
    if n == 0.0 {
        return PerformanceMetrics {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            sortino_ratio: 0.0,
            num_trades,
            win_rate: 0.0,
        };
    }

    // Total return (compounded)
    let total_return = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

    // Sharpe ratio
    let mean = returns.iter().sum::<f64>() / n;
    let variance = if n > 1.0 {
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();
    let sharpe_ratio = if std_dev > 1e-10 {
        mean / std_dev
    } else {
        0.0
    };

    // Max drawdown
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    let mut cumulative = 1.0;
    for r in returns {
        cumulative *= 1.0 + r;
        peak = peak.max(cumulative);
        let dd = (peak - cumulative) / peak;
        max_dd = max_dd.max(dd);
    }

    // Sortino ratio (downside deviation)
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    let downside_variance = if downside_returns.len() > 1 {
        let n_down = downside_returns.len() as f64;
        downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / n_down
    } else {
        0.0
    };
    let downside_dev = downside_variance.sqrt();
    let sortino_ratio = if downside_dev > 1e-10 {
        mean / downside_dev
    } else {
        0.0
    };

    // Win rate
    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    let win_rate = if !returns.is_empty() {
        wins as f64 / returns.len() as f64
    } else {
        0.0
    };

    PerformanceMetrics {
        total_return,
        sharpe_ratio,
        max_drawdown: max_dd,
        sortino_ratio,
        num_trades,
        win_rate,
    }
}

/// Train a Q-learning agent on the environment with the given reward shaper.
pub fn train_agent(
    agent: &mut QLearningAgent,
    candles: &[Candle],
    shaper: &mut dyn RewardShaper,
    num_episodes: usize,
    initial_balance: f64,
) -> Vec<f64> {
    let mut episode_returns = Vec::new();

    for _ in 0..num_episodes {
        let mut env = TradingEnvironment::new(candles.to_vec(), initial_balance);
        let mut state = env.reset();
        shaper.reset();

        loop {
            let s = state.discretize();
            let action = agent.select_action(s);
            let prev_state = state.clone();
            let (next_state, _base_reward, done) = env.step(action);

            if done {
                break;
            }

            let shaped_reward = shaper.shape(&prev_state, action, &next_state);
            let ns = next_state.discretize();
            agent.update(s, action.index(), shaped_reward, ns);
            state = next_state;
        }

        agent.decay_epsilon();
        let final_return = (state.portfolio_value - initial_balance) / initial_balance;
        episode_returns.push(final_return);
    }

    episode_returns
}

/// Evaluate a trained agent on candle data (greedy policy).
pub fn evaluate_agent(
    agent: &QLearningAgent,
    candles: &[Candle],
    initial_balance: f64,
) -> (PerformanceMetrics, Vec<f64>) {
    let mut env = TradingEnvironment::new(candles.to_vec(), initial_balance);
    let mut state = env.reset();
    let mut num_trades = 0usize;

    loop {
        let s = state.discretize();
        let action = agent.best_action(s);
        let prev_position = state.position;
        let (next_state, _, done) = env.step(action);

        if done {
            break;
        }

        if (next_state.position - prev_position).abs() > 0.5 {
            num_trades += 1;
        }
        state = next_state;
    }

    let returns = &state.returns_history;
    let metrics = compute_metrics(returns, num_trades);
    (metrics, returns.clone())
}

// ============================================================================
// Bybit API Client
// ============================================================================

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Client for fetching market data from Bybit.
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first; reverse for chronological order
        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic candle data for testing.
pub fn generate_synthetic_candles(num_candles: usize, start_price: f64, seed: u64) -> Vec<Candle> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut price = start_price;
    let mut candles = Vec::with_capacity(num_candles);

    for i in 0..num_candles {
        let ret: f64 = rng.gen_range(-0.03..0.03);
        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1000000 + i as u64 * 60000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnl_reward_basic() {
        let reward = PnlReward::compute(1.0, 105.0, 100.0);
        assert!((reward - 0.05).abs() < 1e-10, "Expected 0.05, got {}", reward);

        let reward_short = PnlReward::compute(-1.0, 105.0, 100.0);
        assert!(
            (reward_short - (-0.05)).abs() < 1e-10,
            "Expected -0.05, got {}",
            reward_short
        );
    }

    #[test]
    fn test_pnl_reward_zero_price() {
        let reward = PnlReward::compute(1.0, 105.0, 0.0);
        assert!((reward - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_rolling() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = SharpeShapingReward::rolling_sharpe(&returns, 5);
        assert!(sharpe > 0.0, "Positive returns should have positive Sharpe");

        let neg_returns = vec![-0.01, -0.02, -0.01, -0.015, -0.005];
        let neg_sharpe = SharpeShapingReward::rolling_sharpe(&neg_returns, 5);
        assert!(
            neg_sharpe < 0.0,
            "Negative returns should have negative Sharpe"
        );
    }

    #[test]
    fn test_sharpe_empty_returns() {
        let sharpe = SharpeShapingReward::rolling_sharpe(&[], 5);
        assert!((sharpe - 0.0).abs() < 1e-10);

        let sharpe_one = SharpeShapingReward::rolling_sharpe(&[0.01], 5);
        assert!((sharpe_one - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_penalty() {
        let mut dd = DrawdownPenaltyReward::new(1.0, 0.99);

        let prev = TradingState {
            step: 0,
            position: 1.0,
            portfolio_value: 100.0,
            peak_value: 100.0,
            returns_history: vec![],
            price: 100.0,
            prev_price: 100.0,
        };

        let curr = TradingState {
            step: 1,
            position: 1.0,
            portfolio_value: 90.0,
            peak_value: 100.0,
            returns_history: vec![-0.1],
            price: 95.0,
            prev_price: 100.0,
        };

        let shaped = dd.shape(&prev, Action::Hold, &curr);
        // Previous potential = 0 (no drawdown), current potential = -0.1 (10% drawdown)
        // F = 0.99 * (-0.1) - 0.0 = -0.099
        assert!(shaped < 0.0, "Drawdown should produce negative shaping: {}", shaped);
    }

    #[test]
    fn test_transaction_cost_penalty() {
        let mut tc = TransactionCostPenalty::new(0.001);

        let state = TradingState {
            step: 0,
            position: 0.0,
            portfolio_value: 100.0,
            peak_value: 100.0,
            returns_history: vec![],
            price: 100.0,
            prev_price: 100.0,
        };

        let next_state = state.clone();

        // Buy from flat: position change = 1.0
        let cost = tc.shape(&state, Action::Buy, &next_state);
        assert!(
            (cost - (-0.001)).abs() < 1e-10,
            "Expected -0.001, got {}",
            cost
        );

        // Hold from flat: no position change
        let no_cost = tc.shape(&state, Action::Hold, &next_state);
        assert!(
            (no_cost - 0.0).abs() < 1e-10,
            "Expected 0, got {}",
            no_cost
        );
    }

    #[test]
    fn test_composite_reward() {
        let mut composite = CompositeReward::new("test_composite");
        composite.add_component(1.0, Box::new(PnlReward::new()));
        composite.add_component(0.5, Box::new(TransactionCostPenalty::new(0.001)));

        let prev = TradingState {
            step: 0,
            position: 1.0,
            portfolio_value: 100.0,
            peak_value: 100.0,
            returns_history: vec![],
            price: 100.0,
            prev_price: 100.0,
        };

        let curr = TradingState {
            step: 1,
            position: 1.0,
            portfolio_value: 105.0,
            peak_value: 105.0,
            returns_history: vec![0.05],
            price: 105.0,
            prev_price: 100.0,
        };

        let reward = composite.shape(&prev, Action::Hold, &curr);
        // PnL: 1.0 * 0.05 = 0.05
        // TC: 0.5 * 0.0 = 0.0 (Hold with same position)
        assert!(
            (reward - 0.05).abs() < 1e-10,
            "Expected 0.05, got {}",
            reward
        );
    }

    #[test]
    fn test_potential_based_shaping_wrapper() {
        let mut shaper = PotentialBasedShaping::new(
            |s: &TradingState| s.portfolio_value / 100.0,
            0.99,
            "value_potential",
        );

        let prev = TradingState {
            step: 0,
            position: 0.0,
            portfolio_value: 100.0,
            peak_value: 100.0,
            returns_history: vec![],
            price: 100.0,
            prev_price: 100.0,
        };

        let curr = TradingState {
            step: 1,
            position: 1.0,
            portfolio_value: 110.0,
            peak_value: 110.0,
            returns_history: vec![0.1],
            price: 110.0,
            prev_price: 100.0,
        };

        let shaped = shaper.shape(&prev, Action::Buy, &curr);
        // gamma * Phi(s') - Phi(s) = 0.99 * 1.1 - 1.0 = 0.089
        let expected = 0.99 * 1.1 - 1.0;
        assert!(
            (shaped - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            shaped
        );
    }

    #[test]
    fn test_q_learning_agent() {
        let mut agent = QLearningAgent::new(0.1, 0.99, 1.0, 0.99, 0.01);

        // Initially all Q-values are 0
        assert!((agent.get_q(0, 0) - 0.0).abs() < 1e-10);

        // Update should change Q-value
        agent.update(0, 0, 1.0, 1);
        assert!(agent.get_q(0, 0) > 0.0);

        // Epsilon decay
        let old_eps = agent.epsilon;
        agent.decay_epsilon();
        assert!(agent.epsilon < old_eps);
    }

    #[test]
    fn test_state_discretize() {
        let state = TradingState {
            step: 0,
            position: 0.0,
            portfolio_value: 100.0,
            peak_value: 100.0,
            returns_history: vec![],
            price: 100.0,
            prev_price: 100.0,
        };

        let bin = state.discretize();
        // position=0 -> bin 1, return=0 -> bin 2, dd=0 -> bin 0
        assert_eq!(bin, 1 * 15 + 2 * 3 + 0);
    }

    #[test]
    fn test_compute_metrics() {
        let returns = vec![0.01, 0.02, -0.005, 0.015, 0.01];
        let metrics = compute_metrics(&returns, 5);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe_ratio > 0.0);
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.win_rate > 0.5);
        assert_eq!(metrics.num_trades, 5);
    }

    #[test]
    fn test_synthetic_candles() {
        let candles = generate_synthetic_candles(100, 50000.0, 42);
        assert_eq!(candles.len(), 100);
        assert!((candles[0].open - 50000.0).abs() < 1e-6);
        for c in &candles {
            assert!(c.high >= c.open.max(c.close));
            assert!(c.low <= c.open.min(c.close));
            assert!(c.volume > 0.0);
        }
    }

    #[test]
    fn test_trading_environment() {
        let candles = generate_synthetic_candles(50, 100.0, 123);
        let mut env = TradingEnvironment::new(candles, 10000.0);

        let state = env.reset();
        assert!((state.portfolio_value - 10000.0).abs() < 1e-10);
        assert!((state.position - 0.0).abs() < 1e-10);

        let (next_state, _reward, done) = env.step(Action::Buy);
        assert!(!done);
        assert!((next_state.position - 1.0).abs() < 1e-10);
    }
}
