use reward_shaping_trading::*;

fn main() {
    println!("=== Chapter 309: Reward Shaping for Trading ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch data from Bybit or fall back to synthetic data
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT data from Bybit...");
    let candles = match BybitClient::new().fetch_klines("BTCUSDT", "60", 200) {
        Ok(c) if !c.is_empty() => {
            println!("Fetched {} candles from Bybit.\n", c.len());
            c
        }
        Ok(_) => {
            println!("Empty response from Bybit, using synthetic data.\n");
            generate_synthetic_candles(200, 50000.0, 42)
        }
        Err(e) => {
            println!("Bybit API error: {}. Using synthetic data.\n", e);
            generate_synthetic_candles(200, 50000.0, 42)
        }
    };

    let initial_balance = 10000.0;
    let num_episodes = 500;

    // Split into train and test
    let split = candles.len() * 3 / 4;
    let train_candles = &candles[..split];
    let test_candles = &candles[split..];

    println!(
        "Train candles: {}, Test candles: {}\n",
        train_candles.len(),
        test_candles.len()
    );

    // -----------------------------------------------------------------------
    // 2. Agent 1: Raw PnL reward
    // -----------------------------------------------------------------------
    println!("--- Agent 1: Raw PnL Reward ---");
    let mut agent_pnl = QLearningAgent::new(0.1, 0.99, 1.0, 0.995, 0.01);
    let mut pnl_shaper = PnlReward::new();

    let pnl_returns = train_agent(
        &mut agent_pnl,
        train_candles,
        &mut pnl_shaper,
        num_episodes,
        initial_balance,
    );

    let avg_train_return: f64 =
        pnl_returns.iter().rev().take(50).sum::<f64>() / 50.0_f64.min(pnl_returns.len() as f64);
    println!("  Avg training return (last 50 episodes): {:.4}%", avg_train_return * 100.0);

    let (pnl_metrics, _) = evaluate_agent(&agent_pnl, test_candles, initial_balance);
    println!("  Test: {}\n", pnl_metrics);

    // -----------------------------------------------------------------------
    // 3. Agent 2: Sharpe-shaped reward
    // -----------------------------------------------------------------------
    println!("--- Agent 2: Sharpe-Shaped Reward ---");
    let mut agent_sharpe = QLearningAgent::new(0.1, 0.99, 1.0, 0.995, 0.01);
    let mut sharpe_composite = CompositeReward::new("PnL + Sharpe");
    sharpe_composite.add_component(1.0, Box::new(PnlReward::new()));
    sharpe_composite.add_component(0.5, Box::new(SharpeShapingReward::new(20, 0.1, 0.99)));

    let sharpe_returns = train_agent(
        &mut agent_sharpe,
        train_candles,
        &mut sharpe_composite,
        num_episodes,
        initial_balance,
    );

    let avg_train_sharpe: f64 = sharpe_returns.iter().rev().take(50).sum::<f64>()
        / 50.0_f64.min(sharpe_returns.len() as f64);
    println!(
        "  Avg training return (last 50 episodes): {:.4}%",
        avg_train_sharpe * 100.0
    );

    let (sharpe_metrics, _) = evaluate_agent(&agent_sharpe, test_candles, initial_balance);
    println!("  Test: {}\n", sharpe_metrics);

    // -----------------------------------------------------------------------
    // 4. Agent 3: Drawdown-penalized reward
    // -----------------------------------------------------------------------
    println!("--- Agent 3: Drawdown-Penalized Reward ---");
    let mut agent_dd = QLearningAgent::new(0.1, 0.99, 1.0, 0.995, 0.01);
    let mut dd_composite = CompositeReward::new("PnL + Drawdown + TC");
    dd_composite.add_component(1.0, Box::new(PnlReward::new()));
    dd_composite.add_component(1.0, Box::new(DrawdownPenaltyReward::new(0.5, 0.99)));
    dd_composite.add_component(1.0, Box::new(TransactionCostPenalty::new(0.001)));

    let dd_returns = train_agent(
        &mut agent_dd,
        train_candles,
        &mut dd_composite,
        num_episodes,
        initial_balance,
    );

    let avg_train_dd: f64 =
        dd_returns.iter().rev().take(50).sum::<f64>() / 50.0_f64.min(dd_returns.len() as f64);
    println!(
        "  Avg training return (last 50 episodes): {:.4}%",
        avg_train_dd * 100.0
    );

    let (dd_metrics, _) = evaluate_agent(&agent_dd, test_candles, initial_balance);
    println!("  Test: {}\n", dd_metrics);

    // -----------------------------------------------------------------------
    // 5. Summary comparison
    // -----------------------------------------------------------------------
    println!("=== Comparison Summary ===");
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10}",
        "Agent", "Return%", "Sharpe", "MaxDD%", "Sortino"
    );
    println!("{:-<65}", "");

    for (name, m) in [
        ("Raw PnL", &pnl_metrics),
        ("Sharpe-Shaped", &sharpe_metrics),
        ("DD-Penalized", &dd_metrics),
    ] {
        println!(
            "{:<25} {:>9.4} {:>10.4} {:>9.4} {:>10.4}",
            name,
            m.total_return * 100.0,
            m.sharpe_ratio,
            m.max_drawdown * 100.0,
            m.sortino_ratio,
        );
    }
    println!();

    println!("Reward shaping helps RL agents learn risk-aware trading strategies.");
    println!("Sharpe-shaped agents typically show better risk-adjusted returns,");
    println!("while drawdown-penalized agents exhibit smaller maximum drawdowns.");
}
