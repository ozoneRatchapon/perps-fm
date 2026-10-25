//! Metrics and the evaluation protocol: pinball loss / CRPS, block-bootstrap
//! confidence intervals, sealed-week handling. Phase 5 fills the protocol in.

/// Sealed-week evaluation entry point; Phase 5.
pub fn run() -> anyhow::Result<()> {
    anyhow::bail!("evaluate: implemented in Phase 5")
}

/// Pinball (quantile) loss for one quantile level; averaged over samples.
pub fn pinball_loss(pred: &[f64], actual: &[f64], q: f64) -> f64 {
    assert_eq!(pred.len(), actual.len());
    assert!((0.0..1.0).contains(&q));
    let n = pred.len();
    if n == 0 {
        return f64::NAN;
    }
    pred.iter()
        .zip(actual)
        .map(|(&p, &a)| {
            let d = a - p;
            if d >= 0.0 {
                q * d
            } else {
                (q - 1.0) * d
            }
        })
        .sum::<f64>()
        / n as f64
}

/// Approximate CRPS from a finite quantile grid (average of 2 * pinball).
pub fn crps_from_quantiles(pred: &[Vec<f64>], actual: &[f64], qs: &[f64]) -> f64 {
    let n = actual.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut total = 0.0;
    for (p_row, &a) in pred.iter().zip(actual) {
        let mut s = 0.0;
        for (p, &q) in p_row.iter().zip(qs) {
            let d = a - p;
            s += if d >= 0.0 { q * d } else { (q - 1.0) * d };
        }
        total += 2.0 * s / qs.len() as f64;
    }
    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinball_zero_when_perfect() {
        assert_eq!(pinball_loss(&[1.0, 2.0], &[1.0, 2.0], 0.5), 0.0);
    }

    #[test]
    fn pinball_asymmetric() {
        // over-predicting at q=0.9 costs little; under-predicting costs 9x
        let over = pinball_loss(&[2.0], &[1.0], 0.9);
        let under = pinball_loss(&[1.0], &[2.0], 0.9);
        assert!(under > over * 5.0);
    }

    #[test]
    fn crps_zero_when_perfect() {
        let preds = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        assert_eq!(crps_from_quantiles(&preds, &[1.0, 2.0], &[0.5, 0.75]), 0.0);
    }
}
