//! After-cost performance metrics from an equity curve.

/// Per-minute log returns of the equity curve.
pub fn minute_returns(equity: &[f64]) -> Vec<f64> {
    equity
        .windows(2)
        .map(|w| if w[0] > 0.0 { (w[1] / w[0]).ln() } else { 0.0 })
        .collect()
}

/// Annualized Sharpe assuming 1-minute bars: sqrt(525600) scaling.
pub fn sharpe(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = returns.iter().sum::<f64>() / n as f64;
    let var = returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (n - 1) as f64;
    if var <= 0.0 {
        return 0.0;
    }
    mean / var.sqrt() * (525_600.0f64.sqrt())
}

/// Max drawdown as a fraction (positive number).
pub fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut mdd: f64 = 0.0;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        if peak > 0.0 {
            mdd = mdd.max((peak - e) / peak);
        }
    }
    mdd
}

pub fn total_return(equity: &[f64]) -> f64 {
    match (equity.first(), equity.last()) {
        (Some(&s), Some(&e)) if s > 0.0 => e / s - 1.0,
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharpe_of_constant_growth() {
        let eq: Vec<f64> = (0..=1440)
            .map(|i| 100.0 * (1.0 + 0.0001 * i as f64))
            .collect();
        let r = minute_returns(&eq);
        let s = sharpe(&r);
        assert!(s > 5.0, "steady growth should be high Sharpe, got {s}");
    }

    #[test]
    fn drawdown_simple() {
        let eq = [100.0, 110.0, 90.0, 95.0];
        assert!((max_drawdown(&eq) - (110.0 - 90.0) / 110.0).abs() < 1e-12);
        assert!((total_return(&eq) - (-0.05)).abs() < 1e-12);
    }
}
