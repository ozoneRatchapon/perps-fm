//! Trading policies driven by quantile forecasts.

use super::sim::Action;

pub struct DecisionCtx<'a> {
    pub ts: i64,
    pub close: f64,
    /// Model quantile predictions for fwd_ret_30m at this minute, ascending.
    pub quantiles: &'a [f64],
    pub position: f64,
    pub equity: f64,
}

pub trait Policy {
    fn decide(&mut self, ctx: DecisionCtx) -> Action;
}

/// Threshold policy: enter long when q50 >= min_edge (short when <= -min_edge),
/// hold `horizon` minutes, exit on stop or horizon.
pub struct ThresholdPolicy {
    /// Minimum |q50| (in return units) to open a position.
    pub min_edge: f64,
    /// Stop: exit when unrealized return worse than -stop_ret.
    pub stop_ret: f64,
    /// Exposure magnitude when in a position (fraction of equity, notional).
    pub exposure: f64,
    /// Holding period in minutes.
    pub horizon_min: i64,
    entry_ts: Option<i64>,
    entry_px: f64,
    dir: f64,
}

impl ThresholdPolicy {
    pub fn new(min_edge: f64, stop_ret: f64, exposure: f64, horizon_min: i64) -> Self {
        Self {
            min_edge,
            stop_ret,
            exposure,
            horizon_min,
            entry_ts: None,
            entry_px: 0.0,
            dir: 0.0,
        }
    }
}

impl Policy for ThresholdPolicy {
    fn decide(&mut self, ctx: DecisionCtx) -> Action {
        let q50 = ctx
            .quantiles
            .get(ctx.quantiles.len() / 2)
            .copied()
            .unwrap_or(0.0);

        // exit conditions first
        if let Some(entry_ts) = self.entry_ts {
            let held = (ctx.ts - entry_ts) / 60;
            let unrealized = (ctx.close / self.entry_px - 1.0) * self.dir;
            if held >= self.horizon_min || unrealized <= -self.stop_ret {
                self.entry_ts = None;
                self.dir = 0.0;
                return Action {
                    target_exposure: 0.0,
                };
            }
        }

        // entries
        if self.entry_ts.is_none() {
            if q50 >= self.min_edge {
                self.dir = 1.0;
            } else if q50 <= -self.min_edge {
                self.dir = -1.0;
            }
            if self.dir != 0.0 {
                self.entry_ts = Some(ctx.ts);
                self.entry_px = ctx.close;
                return Action {
                    target_exposure: self.dir * self.exposure,
                };
            }
        }
        Action {
            target_exposure: self.dir * self.exposure,
        }
    }
}
