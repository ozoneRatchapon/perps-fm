//! S4D: diagonal structured state-space layer (Gu et al. 2022), implemented
//! from scratch in candle with complex-diagonal state spaces represented as
//! real (re, im) tensor pairs.
//!
//! Training mode: explicit causal convolution. The kernel
//!   K[k, c] = Re( sum_s C[c,s] * A[c,s]^k * B[c,s] )
//! is materialized per forward, then applied as a sum of shifted inputs
//! (LTI, so one broadcast-mul-add per kernel tap). Kernel taps are truncated
//! at `kernel_len` (decays make far taps ~0 at init; documented approximation).
//!
//! Stability: |A| = exp(-softplus(lr_raw)) <= 1, so the recurrence cannot
//! blow up. Init is S4D-LegS-like: |A| = 1/2, frequencies spread over [0, pi).

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder, VarMap};

#[derive(Debug, Clone)]
pub struct S4dConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub seq_len: usize,
    /// Truncation length of the explicit kernel (<= seq_len).
    pub kernel_len: usize,
}

impl Default for S4dConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            d_state: 8,
            seq_len: 256,
            kernel_len: 128,
        }
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // log(1 + exp(x)), numerically stable
    let one = x.ones_like()?;
    let exp_neg = x.neg()?.exp()?;
    x.maximum(&(&one * 0.0)?)? + (&one + exp_neg)?.log()?
}

pub struct S4dBlock {
    cfg: S4dConfig,
    /// |A| = exp(-softplus(lr_raw)) and arg A = ltheta, both (C, S)
    lr_raw: candle_core::Tensor,
    ltheta: candle_core::Tensor,
    b_re: candle_core::Tensor,
    b_im: candle_core::Tensor,
    c_re: candle_core::Tensor,
    c_im: candle_core::Tensor,
    d_skip: candle_core::Tensor, // (C,)
    in_proj: Linear,
    out_proj: Linear,
}

impl S4dBlock {
    /// Seed a fresh VarMap with the S4D initial tensors. Call once before
    /// `new` so `vs.get` finds them.
    pub fn init_vars(cfg: &S4dConfig, varmap: &VarMap, dev: &candle_core::Device) -> Result<()> {
        let (c, s) = (cfg.d_model, cfg.d_state);
        let mut map = varmap.data().lock().unwrap();
        let mut put = |name: &str, t: Tensor| -> Result<()> {
            map.insert(name.to_string(), candle_core::Var::from_tensor(&t)?);
            Ok(())
        };
        put("lr_raw", Tensor::zeros((c, s), DType::F32, dev)?)?;
        let theta: Vec<f32> = (0..c * s)
            .map(|i| (std::f64::consts::PI * (i % s) as f64 / s as f64) as f32)
            .collect();
        put("ltheta", Tensor::from_vec(theta, (c, s), dev)?)?;

        // deterministic xorshift for reproducible B/C init
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        };
        let scale = 1.0 / (s as f64).sqrt() as f32;
        let mut b = Vec::with_capacity(c * s);
        let mut cc = Vec::with_capacity(c * s);
        for _ in 0..c * s {
            b.push(next() * scale);
            cc.push(next() * scale);
        }
        put("b_re", Tensor::from_vec(b, (c, s), dev)?)?;
        put("b_im", Tensor::zeros((c, s), DType::F32, dev)?)?;
        put("c_re", Tensor::from_vec(cc, (c, s), dev)?)?;
        put("c_im", Tensor::zeros((c, s), DType::F32, dev)?)?;
        put("d_skip", Tensor::ones((c,), DType::F32, dev)?)?;
        Ok(())
    }

    pub fn new(cfg: S4dConfig, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            lr_raw: vs.get((cfg.d_model, cfg.d_state), "lr_raw")?,
            ltheta: vs.get((cfg.d_model, cfg.d_state), "ltheta")?,
            b_re: vs.get((cfg.d_model, cfg.d_state), "b_re")?,
            b_im: vs.get((cfg.d_model, cfg.d_state), "b_im")?,
            c_re: vs.get((cfg.d_model, cfg.d_state), "c_re")?,
            c_im: vs.get((cfg.d_model, cfg.d_state), "c_im")?,
            d_skip: vs.get((cfg.d_model,), "d_skip")?,
            in_proj: linear(cfg.d_model, cfg.d_model, vs.pp("in_proj"))?,
            out_proj: linear(cfg.d_model, cfg.d_model, vs.pp("out_proj"))?,
            cfg,
        })
    }

    /// Causal SSM kernel, shape (kernel_len, C).
    pub fn kernel(&self) -> Result<Tensor> {
        let klen = self.cfg.kernel_len;
        let (c, _s) = (self.cfg.d_model, self.cfg.d_state);
        let lr = softplus(&self.lr_raw)?.neg()?; // <= 0, (C,S)
                                                 // cb = C * B (complex product), (C,S)
        let cb_re =
            (self.c_re.broadcast_mul(&self.b_re)? - self.c_im.broadcast_mul(&self.b_im)?)?;
        let cb_im =
            (self.c_re.broadcast_mul(&self.b_im)? + self.c_im.broadcast_mul(&self.b_re)?)?;

        let mut ks = Vec::with_capacity(klen);
        for k in 0..klen {
            let kf = k as f64;
            let decay = (&lr * kf)?.exp()?; // (C,S)
            let ang = (&self.ltheta * kf)?;
            let cos = ang.cos()?;
            let sin = ang.sin()?;
            let re = (cb_re.broadcast_mul(&cos)? - cb_im.broadcast_mul(&sin)?)?;
            let re = re.broadcast_mul(&decay)?;
            ks.push(re.sum(D::Minus1)?.reshape((1, c))?);
        }
        Tensor::cat(&ks, 0) // (klen, C)
    }

    /// x: (B, T, C) -> (B, T, C). T must equal cfg.seq_len.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, t, _c) = x.dims3()?;
        debug_assert_eq!(t, self.cfg.seq_len, "S4D requires fixed seq_len");
        let u = x.apply(&self.in_proj)?.transpose(1, 2)?.contiguous()?; // (B,C,T)

        let k = self.kernel()?; // (klen, C)
                                // y[t] = sum_k K[k,c] * u[t-k] + D[c]*u[t]
        let mut y = u.broadcast_mul(&k.narrow(0, 0, 1)?.transpose(0, 1)?.unsqueeze(0)?)?;
        for kk in 1..self.cfg.kernel_len {
            let shifted = u.pad_with_zeros(2, kk, 0)?.narrow(2, 0, t)?; // causal shift
            let k_row = k.narrow(0, kk, 1)?.transpose(0, 1)?.unsqueeze(0)?; // (1,C,1)
            y = (y + shifted.broadcast_mul(&k_row)?)?;
        }
        y = (y + u.broadcast_mul(&self.d_skip.unsqueeze(0)?.unsqueeze(D::Minus1)?)?)?;

        let y = y.transpose(1, 2)?.contiguous()?; // (B,T,C)
        x + y.apply(&self.out_proj)?
    }
}

/// S4D network: input projection -> stacked S4D blocks -> heads, matching
/// the transformer's I/O contract (same head layout, same param budget goal).
pub struct Net {
    proj_in: candle_nn::Linear,
    market_emb: candle_nn::Embedding,
    blocks: Vec<S4dBlock>,
    ln_f: candle_nn::LayerNorm,
    head_binary: candle_nn::Linear,
    head_loc: candle_nn::Linear,
    head_quant: candle_nn::Linear,
    pub n_params_estimate: usize,
}

impl Net {
    pub fn init_vars(
        cfg: &S4dConfig,
        n_markets: usize,
        n_features: usize,
        n_blocks: usize,
        varmap: &VarMap,
        dev: &candle_core::Device,
    ) -> Result<()> {
        S4dBlock::init_vars(cfg, varmap, dev)?;
        // heads/emb/proj are created through VarBuilder with default inits,
        // which registers them in the VarMap on first access.
        let _ = (n_markets, n_features, n_blocks);
        Ok(())
    }

    pub fn new(
        cfg: &S4dConfig,
        n_markets: usize,
        n_features: usize,
        n_blocks: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        use candle_nn::{embedding, layer_norm, linear};
        let d = cfg.d_model;
        let blocks = (0..n_blocks)
            .map(|i| S4dBlock::new(cfg.clone(), vs.pp(format!("block_{i}"))))
            .collect::<Result<Vec<_>>>()?;
        // estimate params: ssm (4*(C*S) + C) per block + linears
        let per_block = 4 * d * cfg.d_state + d + 2 * d * d;
        Ok(Self {
            proj_in: linear(n_features, d, vs.pp("proj_in"))?,
            market_emb: candle_nn::embedding(n_markets, d, vs.pp("market_emb"))?,
            blocks,
            ln_f: layer_norm(d, 1e-5, vs.pp("ln_f"))?,
            head_binary: linear(d, 2, vs.pp("head_binary"))?,
            head_loc: linear(d, 1, vs.pp("head_loc"))?,
            head_quant: linear(d, crate::QUANTILES.len(), vs.pp("head_quant"))?,
            n_params_estimate: n_blocks * per_block
                + d * n_features
                + d * n_markets
                + d * (2 + 1 + crate::QUANTILES.len()),
        })
    }
}

impl crate::QuantileNet for Net {
    fn forward(
        &self,
        x: &candle_core::Tensor,
        market_ids: &candle_core::Tensor,
    ) -> Result<crate::transformer::TransformerOutput> {
        use candle_nn::Module;
        let mut h = x.apply(&self.proj_in)?;
        let emb = market_ids.apply(&self.market_emb)?.unsqueeze(1)?;
        h = h.broadcast_add(&emb)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        let last = h
            .narrow(1, h.dim(1)? - 1, 1)?
            .squeeze(1)?
            .apply(&self.ln_f)?;
        Ok(crate::transformer::TransformerOutput {
            binary_logits: last.apply(&self.head_binary)?,
            ret_loc: last
                .apply(&self.head_loc)?
                .squeeze(candle_core::D::Minus1)?,
            ret_quantiles: last.apply(&self.head_quant)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn kernel_of_unit_ssm_is_geometric() -> Result<()> {
        // B=C=1, theta=0, lr_raw=0 -> softplus(0)=ln 2 -> |A| = 0.5
        // => K[k] = 0.5^k
        let dev = Device::Cpu;
        let cfg = S4dConfig {
            d_model: 1,
            d_state: 1,
            seq_len: 8,
            kernel_len: 8,
        };
        let mut vm = VarMap::new();
        S4dBlock::init_vars(&cfg, &vm, &dev)?;
        {
            let mut m = vm.data().lock().unwrap();
            m.insert(
                "lr_raw".to_string(),
                candle_core::Var::from_tensor(&Tensor::from_vec(vec![0.0f32], (1, 1), &dev)?)?,
            );
            m.insert(
                "b_re".to_string(),
                candle_core::Var::from_tensor(&Tensor::from_vec(vec![1.0f32], (1, 1), &dev)?)?,
            );
            m.insert(
                "c_re".to_string(),
                candle_core::Var::from_tensor(&Tensor::from_vec(vec![1.0f32], (1, 1), &dev)?)?,
            );
        }
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let blk = S4dBlock::new(cfg, vs)?;
        let k = blk.kernel()?.to_vec2::<f32>()?;
        for (i, row) in k.iter().enumerate() {
            let expect = 0.5f64.powi(i as i32);
            assert!(
                (row[0] as f64 - expect).abs() < 1e-5,
                "k[{i}]={} vs {expect}",
                row[0]
            );
        }
        Ok(())
    }

    #[test]
    fn forward_shape_and_residual() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = S4dConfig {
            d_model: 16,
            d_state: 4,
            seq_len: 32,
            kernel_len: 16,
        };
        let mut vm = VarMap::new();
        S4dBlock::init_vars(&cfg, &vm, &dev)?;
        let vs = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let blk = S4dBlock::new(cfg, vs)?;
        let x = Tensor::randn(0f32, 1f32, (2, 32, 16), &dev)?;
        let y = blk.forward(&x)?;
        assert_eq!(y.dims(), &[2, 32, 16]);
        // sanity: finite, not exploded (kernel test pins exact math)
        let diff = (y - &x)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(diff.is_finite() && diff < 10.0, "output sane, diff={diff}");
        Ok(())
    }
}
