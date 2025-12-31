//! WeDLM Attention with Grouped Query Attention (GQA) and QK-Normalization
//!
//! CRITICAL implementation details:
//! 1. RMSNorm must compute in f32 for numerical stability
//! 2. QK-norm order: Project -> Reshape -> QK-NORM -> Transpose -> RoPE
//! 3. GQA: 32 query heads, 8 KV heads -> repeat KV 4 times

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::WeDLMConfig;

/// RMSNorm (Root Mean Square Layer Normalization)
///
/// Computes: x * rsqrt(mean(x^2) + eps) * weight
/// CRITICAL: Must compute variance in f32 for numerical stability
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    /// Load from existing tensor (for testing)
    pub fn from_tensor(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();

        // CRITICAL: Compute in f32 for numerical stability
        let x_f32 = x.to_dtype(DType::F32)?;

        // variance = mean(x^2)
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;

        // x_normed = x * rsqrt(variance + eps)
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // Scale by weight and cast back to original dtype
        let result = x_normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        result.to_dtype(input_dtype)
    }
}

/// Grouped Query Attention with optional QK-Normalization
pub struct WeDLMAttention {
    /// Query projection: [hidden_size -> num_heads * head_dim]
    q_proj: Linear,
    /// Key projection: [hidden_size -> num_kv_heads * head_dim]
    k_proj: Linear,
    /// Value projection: [hidden_size -> num_kv_heads * head_dim]
    v_proj: Linear,
    /// Output projection: [num_heads * head_dim -> hidden_size]
    o_proj: Linear,

    /// Query normalization (if qk_norm=true)
    q_norm: Option<RMSNorm>,
    /// Key normalization (if qk_norm=true)
    k_norm: Option<RMSNorm>,

    /// Number of attention heads
    num_heads: usize,
    /// Number of key-value heads
    num_kv_heads: usize,
    /// Per-head dimension
    head_dim: usize,
    /// Number of query heads per KV head
    num_kv_groups: usize,

    /// Scaling factor: 1/sqrt(head_dim)
    scale: f64,
}

impl WeDLMAttention {
    pub fn new(config: &WeDLMConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // Projections without bias (attention_bias = false for WeDLM)
        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        // QK normalization if enabled
        let (q_norm, k_norm) = if config.qk_norm {
            (
                Some(RMSNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?),
                Some(RMSNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        let scale = 1.0 / (head_dim as f64).sqrt();
        let num_kv_groups = num_heads / num_kv_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups,
            scale,
        })
    }

    /// Forward pass through attention
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `attention_mask` - Optional mask [batch, 1, seq_len, kv_len] (additive, -inf for masked)
    /// * `cos` - RoPE cosine values [seq_len, head_dim]
    /// * `sin` - RoPE sine values [seq_len, head_dim]
    /// * `kv_cache` - Optional (cached_k, cached_v) for incremental decoding
    ///
    /// # Returns
    /// (output, new_kv_cache) where output is [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // 1. Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // 2. Reshape to [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        // 3. Apply QK-norm BEFORE transpose, BEFORE RoPE
        // CRITICAL: This order is crucial for WeDLM!
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        // 4. Transpose to [batch, num_heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // 5. Apply RoPE
        let (q, k) = super::rope::RotaryEmbedding::apply(&q, &k, cos, sin)?;

        // 6. Handle KV cache
        let (k, v) = match kv_cache {
            Some((cached_k, cached_v)) => {
                let k = Tensor::cat(&[cached_k, &k], 2)?;
                let v = Tensor::cat(&[cached_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };

        // Store KV for cache
        let new_kv = (k.clone(), v.clone());

        // 7. Attention computation
        // Use fused SDPA on Metal/CUDA, fall back to manual on CPU
        let attn_output = if q.device().is_cpu() {
            // CPU fallback: manual attention (SDPA has no CPU impl)
            let k_expanded = self.repeat_kv(&k)?;
            let v_expanded = self.repeat_kv(&v)?;
            let attn_weights = (q.matmul(&k_expanded.transpose(2, 3)?)? * self.scale)?;
            let attn_weights = match attention_mask {
                Some(mask) => attn_weights.broadcast_add(mask)?,
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
            attn_weights.matmul(&v_expanded)?
        } else {
            // GPU: Fused SDPA kernel
            // - Handles GQA natively (no repeat_kv needed)
            // - Fuses: matmul, scale, mask, softmax, matmul into single kernel
            // - Uses Metal-optimized SDPA kernels
            //
            // SDPA requires mask shape [bs, num_heads, qseq, kseq]
            let kv_seq = k.dim(2)?;
            let expanded_mask = match attention_mask {
                Some(mask) => Some(mask.broadcast_as((batch_size, self.num_heads, seq_len, kv_seq))?),
                None => None,
            };

            candle_nn::ops::sdpa(
                &q,                           // [batch, num_heads, seq, head_dim]
                &k,                           // [batch, num_kv_heads, kv_seq, head_dim]
                &v,                           // [batch, num_kv_heads, kv_seq, head_dim]
                expanded_mask.as_ref(),       // Expanded mask [batch, num_heads, qseq, kseq]
                false,                        // do_causal=false since we provide explicit mask
                self.scale as f32,            // 1/sqrt(head_dim)
                1.0,                          // softcapping=1.0 means disabled
            )?
        };

        // 8. Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // 13. Output projection
        let output = self.o_proj.forward(&attn_output)?;

        Ok((output, new_kv))
    }

    /// Repeat KV heads for grouped query attention (used in CPU fallback)
    ///
    /// Input: [batch, num_kv_heads, seq, head_dim]
    /// Output: [batch, num_heads, seq, head_dim]
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 {
            return Ok(x.clone());
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;

        // Expand and reshape: [b, kv_heads, 1, seq, dim] -> [b, kv_heads, groups, seq, dim] -> [b, heads, seq, dim]
        x.unsqueeze(2)?
            .expand((batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim))?
            .reshape((batch, self.num_heads, seq_len, head_dim))
    }
}

/// Create a causal attention mask
///
/// Returns an additive mask where future positions are -inf
pub fn create_causal_mask(
    seq_len: usize,
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    // Create mask where position i can attend to positions 0..=i+offset
    // offset = kv_len - seq_len (for when we have cached KV)
    let offset = kv_len as i64 - seq_len as i64;

    let mut mask_data = vec![0.0f32; seq_len * kv_len];
    for i in 0..seq_len {
        for j in 0..kv_len {
            // Position i (in query) can attend to position j (in key) if j <= i + offset
            if j as i64 > i as i64 + offset {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (seq_len, kv_len), device)?.to_dtype(dtype)?;

    // Reshape for broadcasting: [1, 1, seq_len, kv_len]
    mask.unsqueeze(0)?.unsqueeze(0)
}

/// Create a WeDLM attention mask for parallel decoding
///
/// After topological reordering, the sequence is [filled tokens..., MASK tokens...].
/// - Filled tokens attend causally to each other
/// - MASK tokens attend to ALL filled tokens but NOT to each other
///
/// # Arguments
/// * `seq_len` - Length of query sequence (block tokens only)
/// * `kv_len` - Length of key sequence (prefix + block)
/// * `num_filled_in_block` - Number of filled (non-MASK) tokens in the block
/// * `dtype` - Data type for the mask
/// * `device` - Device to place the mask on
pub fn create_wedlm_mask(
    seq_len: usize,
    kv_len: usize,
    num_filled_in_block: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    // Query positions [0, seq_len) correspond to reordered block tokens
    // KV positions [0, kv_len) correspond to [prefix..., block...]
    // prefix_len = kv_len - seq_len
    let prefix_len = kv_len - seq_len;

    let mut mask_data = vec![0.0f32; seq_len * kv_len];
    for i in 0..seq_len {
        for j in 0..kv_len {
            let is_query_mask = i >= num_filled_in_block;
            let is_key_in_prefix = j < prefix_len;
            let is_key_filled_in_block = j >= prefix_len && j < prefix_len + num_filled_in_block;

            let can_attend = if is_query_mask {
                // MASK token: can only attend to prefix + filled block tokens
                is_key_in_prefix || is_key_filled_in_block
            } else {
                // Filled token: standard causal attention
                // Can attend to prefix + earlier filled tokens in block
                let key_block_idx = j as i64 - prefix_len as i64;
                is_key_in_prefix || (key_block_idx >= 0 && key_block_idx <= i as i64)
            };

            if !can_attend {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (seq_len, kv_len), device)?.to_dtype(dtype)?;
    mask.unsqueeze(0)?.unsqueeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() -> Result<()> {
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu)?;
        let norm = RMSNorm::from_tensor(weight, 1e-6);

        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu)?;
        let result = norm.forward(&x)?;

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
        // Normalized values should be x / rms
        let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;
        assert!(result_vec[0] > 0.0);
        assert!(result_vec[0] < 1.0); // Should be normalized

        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let mask = create_causal_mask(3, 3, DType::F32, &Device::Cpu)?;
        let mask_vec: Vec<f32> = mask.flatten_all()?.to_vec1()?;

        // Lower triangular: positions (0,0), (1,0), (1,1), (2,0), (2,1), (2,2) should be 0
        // Upper triangular: (0,1), (0,2), (1,2) should be -inf
        assert_eq!(mask_vec[0], 0.0); // (0,0)
        assert!(mask_vec[1].is_infinite() && mask_vec[1] < 0.0); // (0,1)
        assert!(mask_vec[2].is_infinite() && mask_vec[2] < 0.0); // (0,2)
        assert_eq!(mask_vec[3], 0.0); // (1,0)
        assert_eq!(mask_vec[4], 0.0); // (1,1)
        assert!(mask_vec[5].is_infinite() && mask_vec[5] < 0.0); // (1,2)

        Ok(())
    }
}
