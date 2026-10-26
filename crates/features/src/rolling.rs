//! Trailing-window statistics with explicit look-ahead semantics.
//!
//! Convention: `stat(x, t, window)` uses `x[t-window+1 ..= t]` — the value at
//! `t` is INCLUDED, nothing beyond `t` is ever touched. Early positions
//! (fewer than `window` samples) return `None`; callers map that to NaN.
//!
//! Hand-rolled instead of polars lazy rolling windows so the semantics are
//! auditable in one place and testable against brute force (see
//! `tests/lookahead.rs`).

/// Running trailing mean over f64 slices, O(n).
pub struct RollingMean {
    window: usize,
    buf: Vec<f64>,
    sum: f64,
    len: usize,
    head: usize,
}

impl RollingMean {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            buf: vec![0.0; window],
            sum: 0.0,
            len: 0,
            head: 0,
        }
    }

    pub fn push(&mut self, v: f64) -> Option<f64> {
        if self.len < self.window {
            self.buf[self.head] = v;
            self.sum += v;
            self.head = (self.head + 1) % self.window;
            self.len += 1;
        } else {
            let old = self.buf[self.head];
            self.sum += v - old;
            self.buf[self.head] = v;
            self.head = (self.head + 1) % self.window;
        }
        if self.len == self.window {
            Some(self.sum / self.window as f64)
        } else {
            None
        }
    }
}

/// Trailing mean/std pair (population std) via Welford-style running sums.
pub struct RollingMeanStd {
    mean: RollingSum,
    sq: RollingSum,
    window: usize,
}

impl RollingMeanStd {
    pub fn new(window: usize) -> Self {
        Self {
            mean: RollingSum::new(window),
            sq: RollingSum::new(window),
            window,
        }
    }

    pub fn push(&mut self, v: f64) -> Option<(f64, f64)> {
        // Both accumulators must receive EVERY value, or their windows drift
        // out of lockstep (sq would fill w-1 pushes late).
        let s = self.mean.push(v);
        let ss = self.sq.push(v * v);
        match (s, ss) {
            (Some(s), Some(ss)) => {
                let m = s / self.window as f64;
                let var = (ss / self.window as f64 - m * m).max(0.0);
                Some((m, var.sqrt()))
            }
            _ => None,
        }
    }
}

/// Trailing sum, O(1) per step.
pub struct RollingSum {
    buf: Vec<f64>,
    sum: f64,
    len: usize,
    head: usize,
    window: usize,
}

impl RollingSum {
    pub fn new(window: usize) -> Self {
        Self {
            buf: vec![0.0; window],
            sum: 0.0,
            len: 0,
            head: 0,
            window,
        }
    }

    pub fn push(&mut self, v: f64) -> Option<f64> {
        if self.len < self.window {
            self.buf[self.head] = v;
            self.sum += v;
            self.head = (self.head + 1) % self.window;
            self.len += 1;
        } else {
            let old = self.buf[self.head];
            self.sum += v - old;
            self.buf[self.head] = v;
            self.head = (self.head + 1) % self.window;
        }
        if self.len == self.window {
            Some(self.sum)
        } else {
            None
        }
    }
}

/// Trailing count of nonzero values + their sum (for sparse event features).
pub struct RollingCountSum {
    sum: RollingSum,
    cnt: Vec<u32>,
    head: usize,
    len: usize,
    window: usize,
    count: u32,
}

impl RollingCountSum {
    pub fn new(window: usize) -> Self {
        Self {
            sum: RollingSum::new(window),
            cnt: vec![0; window],
            head: 0,
            len: 0,
            window,
            count: 0,
        }
    }

    /// `value` is a per-step amount; nonzero steps count as events.
    pub fn push(&mut self, value: f64) -> Option<(u64, f64)> {
        if self.len < self.window {
            self.cnt[self.head] = if value != 0.0 { 1 } else { 0 };
            self.head = (self.head + 1) % self.window;
            self.len += 1;
        } else {
            self.cnt[self.head] = if value != 0.0 { 1 } else { 0 };
            self.head = (self.head + 1) % self.window;
        }
        // window <= 1440: exact fold is cheap and avoids drift
        self.count = self.cnt.iter().sum::<u32>();
        let s = self.sum.push(value)?;
        Some((self.count as u64, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_mean_matches_brute_force() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 1.37).sin()).collect();
        let w = 7;
        let mut rm = RollingMean::new(w);
        for (t, &v) in x.iter().enumerate() {
            let got = rm.push(v);
            if t + 1 < w {
                assert!(got.is_none());
            } else {
                let expect: f64 = x[t + 1 - w..=t].iter().sum::<f64>() / w as f64;
                assert!((got.unwrap() - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn mean_std_matches_brute_force() {
        let x: Vec<f64> = (0..100)
            .map(|i| (i as f64).cbrt() * 3.1 - i as f64 * 0.01)
            .collect();
        let w = 20;
        let mut rs = RollingMeanStd::new(w);
        for (t, &v) in x.iter().enumerate() {
            let got = rs.push(v);
            if t + 1 >= w {
                let sl = &x[t + 1 - w..=t];
                let n = w as f64;
                let m = sl.iter().sum::<f64>() / n;
                let var = sl.iter().map(|s| (s - m) * (s - m)).sum::<f64>() / n;
                let (gm, gsd) =
                    got.unwrap_or_else(|| panic!("mean_std returned None at t={t}, w={w}"));
                assert!((gm - m).abs() < 1e-10);
                assert!((gsd - var.sqrt()).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn count_sum_tracks_events() {
        let x = [0.0, 0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let w = 4;
        let mut rcs = RollingCountSum::new(w);
        for (t, &v) in x.iter().enumerate() {
            let (count, sum) = rcs.push(v).unwrap_or((0, 0.0));
            if t + 1 >= w {
                let sl = &x[t + 1 - w..=t];
                let ec = sl.iter().filter(|&&s| s != 0.0).count() as u64;
                let es: f64 = sl.iter().sum();
                assert_eq!(count, ec, "t={t}");
                assert!((sum - es).abs() < 1e-12, "t={t}");
            }
        }
    }
}
