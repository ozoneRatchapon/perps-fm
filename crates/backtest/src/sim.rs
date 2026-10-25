//! Simulator state: equity, positions, costs.

#[derive(Debug, Clone, Copy)]
pub struct Action {
    /// Target position as a fraction of equity in NOTIONAL terms, in [-1, 1].
    /// 0 = flat. Sign = direction.
    pub target_exposure: f64,
}

#[derive(Debug)]
pub struct SimState {
    /// Cost per notional traded, as a fraction (fee+slippage bps / 10^4).
    pub cost_rate: f64,
    pub equity: f64,
    /// Signed base-asset position.
    pub position_units: f64,
    pub total_cost: f64,
    pub trades: u64,
    pub equity_curve: Vec<f64>,
}

impl SimState {
    pub fn new(cost_bps: f64) -> Self {
        Self {
            cost_rate: cost_bps / 10_000.0,
            equity: 100.0,
            position_units: 0.0,
            total_cost: 0.0,
            trades: 0,
            equity_curve: vec![100.0],
        }
    }

    /// Apply an action at `price` (previous price `prev` marks the last bar).
    pub fn apply(&mut self, action: Action, price: f64, prev: f64) -> anyhow::Result<()> {
        // mark-to-market the existing position over the last bar move
        self.equity += self.position_units * (price - prev);
        let target_units = action.target_exposure * self.equity / price;
        let delta = target_units - self.position_units;
        if delta.abs() * price > 1e-9 {
            let traded_notional = delta.abs() * price;
            let cost = traded_notional * self.cost_rate;
            self.equity -= cost;
            self.total_cost += cost;
            self.trades += 1;
            self.position_units = target_units;
        }
        self.equity_curve.push(self.equity);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_reduce_equity() {
        let mut s = SimState::new(3.0); // 3 bps
        s.apply(
            Action {
                target_exposure: 1.0,
            },
            100.0,
            100.0,
        )
        .unwrap();
        let after_entry = s.equity;
        assert!(after_entry < 100.0);
        assert!((after_entry - (100.0 - 100.0 * 0.0003)).abs() < 1e-6);
        assert_eq!(s.trades, 1);
    }
}
