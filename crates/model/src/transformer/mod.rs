//! Small causal transformer with a quantile head, written from scratch in
//! candle. ~100k params at default config: deliberately tiny so every run
//! trains in minutes on a MacBook (JD: "small transformers", "small, cheap
//! experiments").
//!
//! Input: (B, T, F) float features per minute. Output heads:
//! - binary logits (B, 2): [liq_burst, vol_burst]
//! - point prediction (B,) of the forward return
//! - quantile predictions (B, Q) of the forward return distribution

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, ops, Activation, AdamW, LayerNorm, Linear, Module, Optimizer, ParamsAdamW,
    VarBuilder, VarMap,
};

use crate::QUANTILES;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerConfig {
    pub n_features: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_mult: usize,
    pub n_markets: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            n_features: 12,
            seq_len: 256,
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            ffn_mult: 2,
            n_markets: 4,
        }
    }
}

impl TransformerConfig {
    pub fn n_params_estimate(&self) -> usize {
        let d = self.d_model;
        let attn = 4 * d * d;
        let ffn = 2 * d * d * self.ffn_mult;
        let per_layer = attn + ffn + 4 * d; // + layernorm weights
        let head = d * (2 + 1 + QUANTILES.len());
        self.n_layers * per_layer + d * self.n_features + head + d * self.n_markets
    }
}

struct CausalSelfAttention {
    qkv: Linear,
    proj: Linear,
    n_heads: usize,
    head_dim: usize,
    d3: usize,       // per-projection dim (C)
    neg_inf: Tensor, // (1, 1, T, T) causal mask
}

impl CausalSelfAttention {
    fn new(cfg: &TransformerConfig, vs: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let qkv = linear(d, 3 * d, vs.pp("qkv"))?;
        let proj = linear(d, d, vs.pp("proj"))?;
        let mut mask = vec![0f32; cfg.seq_len * cfg.seq_len];
        for i in 0..cfg.seq_len {
            for j in 0..cfg.seq_len {
                if j > i {
                    mask[i * cfg.seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let neg_inf = Tensor::from_vec(mask, (1, 1, cfg.seq_len, cfg.seq_len), vs.device())?;
        Ok(Self {
            qkv,
            proj,
            n_heads: cfg.n_heads,
            head_dim: d / cfg.n_heads,
            d3: d,
            neg_inf,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;
        let qkv = x.apply(&self.qkv)?; // (B, T, 3C)
        let q = qkv.narrow(D::Minus1, 0, self.d3)?;
        let k = qkv.narrow(D::Minus1, self.d3, self.d3)?;
        let v = qkv.narrow(D::Minus1, 2 * self.d3, self.d3)?;

        let shape = (b, t, self.n_heads, self.head_dim);
        let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let scores = scores.broadcast_add(&self.neg_inf)?;
        let attn = ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?;
        let out = out.reshape((b, t, self.n_heads * self.head_dim))?;
        out.apply(&self.proj)
    }
}

struct Block {
    attn: CausalSelfAttention,
    ln1: LayerNorm,
    ln2: LayerNorm,
    ffn_in: Linear,
    ffn_out: Linear,
}

impl Block {
    fn new(cfg: &TransformerConfig, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: CausalSelfAttention::new(cfg, vs.pp("attn"))?,
            ln1: layer_norm(cfg.d_model, 1e-5, vs.pp("ln1"))?,
            ln2: layer_norm(cfg.d_model, 1e-5, vs.pp("ln2"))?,
            ffn_in: linear(cfg.d_model, cfg.d_model * cfg.ffn_mult, vs.pp("ffn_in"))?,
            ffn_out: linear(cfg.d_model * cfg.ffn_mult, cfg.d_model, vs.pp("ffn_out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.apply(&self.ln1)?;
        let x = (x + self.attn.forward(&h)?)?;
        let h = x.apply(&self.ln2)?;
        let h = h.apply(&self.ffn_in)?.apply(&Activation::Gelu)?;
        x + h.apply(&self.ffn_out)?
    }
}

pub struct TinyTransformer {
    proj_in: Linear,
    market_emb: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head_binary: Linear, // liq burst, vol burst
    head_loc: Linear,    // central tendency of fwd return
    head_quant: Linear,  // Q quantiles of fwd return
}

pub struct TransformerOutput {
    /// (B, 2) logits: [liq_burst, vol_burst]
    pub binary_logits: Tensor,
    /// (B,) point prediction of fwd return
    pub ret_loc: Tensor,
    /// (B, Q) quantile predictions of fwd return, ascending q
    pub ret_quantiles: Tensor,
}

impl TinyTransformer {
    pub fn new(cfg: &TransformerConfig, vs: VarBuilder) -> Result<Self> {
        let blocks = (0..cfg.n_layers)
            .map(|i| Block::new(cfg, vs.pp(format!("block_{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_in: linear(cfg.n_features, cfg.d_model, vs.pp("proj_in"))?,
            market_emb: candle_nn::embedding(cfg.n_markets, cfg.d_model, vs.pp("market_emb"))?,
            blocks,
            ln_f: layer_norm(cfg.d_model, 1e-5, vs.pp("ln_f"))?,
            head_binary: linear(cfg.d_model, 2, vs.pp("head_binary"))?,
            head_loc: linear(cfg.d_model, 1, vs.pp("head_loc"))?,
            head_quant: linear(cfg.d_model, QUANTILES.len(), vs.pp("head_quant"))?,
        })
    }

    /// x: (B, T, F), market_ids: (B,) i64
    pub fn forward(&self, x: &Tensor, market_ids: &Tensor) -> Result<TransformerOutput> {
        let mut h = x.apply(&self.proj_in)?;
        let emb = market_ids.apply(&self.market_emb)?; // (B, C)
        let emb = emb.unsqueeze(1)?; // (B, 1, C) broadcast over time
        h = h.broadcast_add(&emb)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        let last = h
            .narrow(1, h.dim(1)? - 1, 1)?
            .squeeze(1)?
            .apply(&self.ln_f)?;
        Ok(TransformerOutput {
            binary_logits: last.apply(&self.head_binary)?,
            ret_loc: last.apply(&self.head_loc)?.squeeze(D::Minus1)?,
            ret_quantiles: last.apply(&self.head_quant)?,
        })
    }
}

/// Pinball (quantile) loss on a batch. pred: (B, Q), target: (B,), qs: Q levels.
pub fn quantile_loss(pred: &Tensor, target: &Tensor, qs: &[f64]) -> Result<Tensor> {
    let device = pred.device();
    let q = Tensor::from_slice(qs, (1, qs.len()), device)?.to_dtype(pred.dtype())?;
    let t = target.unsqueeze(D::Minus1)?; // (B, 1)
    let d = t.broadcast_sub(pred)?; // (B, Q)
    let qd = q.broadcast_mul(&d)?;
    let qm1d = ((q - 1.0)?).broadcast_mul(&d)?;
    qd.maximum(&qm1d)?.mean_all()
}

/// AdamW with modest weight decay.
pub fn optimizer(vars: Vec<candle_core::Var>, lr: f64) -> Result<AdamW> {
    Ok(AdamW::new(
        vars,
        ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        },
    )?)
}

/// Allocate the model + optimizer on a device.
pub fn build(cfg: &TransformerConfig, dev: &Device) -> Result<(TinyTransformer, VarMap, AdamW)> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = TinyTransformer::new(cfg, vs)?;
    let opt = optimizer(varmap.all_vars(), 3e-4)?;
    Ok((model, varmap, opt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shapes_and_loss() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = TransformerConfig {
            seq_len: 32,
            n_features: 4,
            ..Default::default()
        };
        let (model, _vm, _opt) = build(&cfg, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (2, 32, 4), &dev)?;
        let mids = Tensor::from_vec(vec![0i64, 3], (2,), &dev)?;
        let out = model.forward(&x, &mids)?;
        assert_eq!(out.binary_logits.dims(), &[2, 2]);
        assert_eq!(out.ret_loc.dims(), &[2]);
        assert_eq!(out.ret_quantiles.dims(), &[2, QUANTILES.len()]);

        let target = Tensor::randn(0f32, 0.01f32, (2,), &dev)?;
        let loss = quantile_loss(&out.ret_quantiles, &target, &QUANTILES)?;
        let l: f32 = loss.to_scalar()?;
        assert!(l.is_finite() && l >= 0.0);
        Ok(())
    }

    #[test]
    fn param_count_small() {
        let cfg = TransformerConfig::default();
        let n = cfg.n_params_estimate();
        assert!(n > 10_000 && n < 1_000_000, "params={n}");
    }

    #[test]
    fn quantile_loss_asymmetric() -> Result<()> {
        let dev = Device::Cpu;
        let pred = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &dev)?;
        let above = Tensor::from_vec(vec![1.0f32], (1,), &dev)?;
        let below = Tensor::from_vec(vec![-1.0f32], (1,), &dev)?;
        let qs = [0.9, 0.5];
        let l_above: f32 = quantile_loss(&pred, &above, &qs)?.to_scalar()?;
        let l_below: f32 = quantile_loss(&pred, &below, &qs)?.to_scalar()?;
        assert!(l_above > l_below);
        Ok(())
    }
}
