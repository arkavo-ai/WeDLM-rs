//! Rotary Position Embeddings (RoPE) for WeDLM
//!
//! CRITICAL: This implementation must match the Python exactly.
//! - theta = 1,000,000.0 (WeDLM uses high theta for long context)
//! - head_dim = 128 -> 64 inverse frequencies
//! - Formula: inv_freq[i] = 1.0 / (theta ^ (2*i / head_dim))
//!
//! Uses candle_nn's fused RoPE kernel for Metal acceleration.

use candle_core::{DType, Device, Result, Tensor};

/// Precomputed rotary position embeddings
pub struct RotaryEmbedding {
    /// Inverse frequencies: [head_dim / 2]
    #[allow(dead_code)]
    inv_freq: Tensor,
    /// Head dimension
    #[allow(dead_code)]
    head_dim: usize,
    /// Maximum sequence length for cached values
    max_seq_len: usize,
    /// Cached cosine values: [max_seq_len, head_dim]
    cos_cache: Tensor,
    /// Cached sine values: [max_seq_len, head_dim]
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    /// Create rotary embeddings with given parameters
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per attention head (128 for WeDLM)
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `theta` - RoPE base frequency (1,000,000 for WeDLM)
    /// * `dtype` - Data type for cos/sin tensors
    /// * `device` - Device to place tensors on
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies: theta^(-2i/d) for i in [0, half_dim)
        // CRITICAL: Must match Python exactly
        // inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = (2 * i) as f64 / head_dim as f64;
                1.0 / theta.powf(exponent) as f32
            })
            .collect();

        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

        // Precompute cos/sin for all positions up to max_seq_len
        // Shape: [max_seq_len, head_dim/2] for fused kernel
        let (cos_cache, sin_cache) =
            Self::compute_cos_sin(&inv_freq, max_seq_len, dtype, device)?;

        Ok(Self {
            inv_freq,
            head_dim,
            max_seq_len,
            cos_cache,
            sin_cache,
        })
    }

    /// Compute cosine and sine values for a range of positions
    ///
    /// Stores cos/sin with shape [seq_len, head_dim/2] for use with
    /// candle_nn's fused RoPE kernel.
    fn compute_cos_sin(
        inv_freq: &Tensor,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // Position indices: [0, 1, 2, ..., seq_len-1]
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;

        // inv_freq shape: [half_dim] -> [1, half_dim]
        let inv_freq = inv_freq.unsqueeze(0)?;

        // freqs = positions @ inv_freq.T -> [seq_len, half_dim]
        let freqs = positions.matmul(&inv_freq)?;

        // Compute cos and sin - keep as [seq_len, half_dim] for fused kernel
        // Store in f32 for numerical stability during computation
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok((cos, sin))
    }

    /// Get cos/sin for sequential positions starting at offset
    ///
    /// # Arguments
    /// * `seq_len` - Length of current sequence
    /// * `offset` - Starting position offset
    ///
    /// # Returns
    /// (cos, sin) tensors of shape [seq_len, head_dim]
    pub fn get_cos_sin(&self, seq_len: usize, offset: usize) -> Result<(Tensor, Tensor)> {
        if offset + seq_len > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "Position {} + {} exceeds max_seq_len {}",
                offset, seq_len, self.max_seq_len
            )));
        }

        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;

        Ok((cos, sin))
    }

    /// Get cos/sin for explicit position indices (for WeDLM parallel decoding)
    ///
    /// This allows non-sequential positions, e.g., for topologically reordered tokens
    /// where token at sequence index 0 might have position 5.
    ///
    /// # Arguments
    /// * `positions` - Tensor of position indices [seq_len] or [batch, seq_len]
    ///
    /// # Returns
    /// (cos, sin) tensors with shape matching positions + [head_dim]
    pub fn get_cos_sin_for_positions(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        // Flatten positions if needed
        let pos_1d = if positions.dims().len() == 2 {
            positions.squeeze(0)?
        } else {
            positions.clone()
        };

        // Use index_select to gather cos/sin for specific positions
        let pos_u32 = pos_1d.to_dtype(DType::U32)?;
        let cos = self.cos_cache.index_select(&pos_u32, 0)?;
        let sin = self.sin_cache.index_select(&pos_u32, 0)?;

        Ok((cos, sin))
    }

    /// Apply rotary embeddings to query and key tensors using fused kernel
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
    /// * `cos` - Cosine values [seq_len, head_dim/2]
    /// * `sin` - Sine values [seq_len, head_dim/2]
    ///
    /// # Returns
    /// (q_rotated, k_rotated) with same shapes as input
    pub fn apply(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Fused kernel requires matching dtypes and contiguous tensors
        let q_dtype = q.dtype();
        let k_dtype = k.dtype();

        // Convert cos/sin to match input dtype for fused kernel
        let cos_q = cos.to_dtype(q_dtype)?;
        let sin_q = sin.to_dtype(q_dtype)?;

        // Make inputs contiguous if needed (fused kernel requirement)
        let q_contig = if q.is_contiguous() {
            q.clone()
        } else {
            q.contiguous()?
        };
        let k_contig = if k.is_contiguous() {
            k.clone()
        } else {
            k.contiguous()?
        };

        // Use candle_nn's fused RoPE kernel (Metal accelerated)
        let q_rotated = candle_nn::rotary_emb::rope(&q_contig, &cos_q, &sin_q)?;

        // For k, convert cos/sin if dtypes differ (unlikely but safe)
        let k_rotated = if k_dtype == q_dtype {
            candle_nn::rotary_emb::rope(&k_contig, &cos_q, &sin_q)?
        } else {
            let cos_k = cos.to_dtype(k_dtype)?;
            let sin_k = sin.to_dtype(k_dtype)?;
            candle_nn::rotary_emb::rope(&k_contig, &cos_k, &sin_k)?
        };

        Ok((q_rotated, k_rotated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv_freq_values() -> Result<()> {
        let rope = RotaryEmbedding::new(128, 16, 1_000_000.0, DType::F32, &Device::Cpu)?;

        // First inv_freq should be 1.0 (theta^0 = 1)
        let inv_freq_vec: Vec<f32> = rope.inv_freq.to_vec1()?;
        assert!((inv_freq_vec[0] - 1.0).abs() < 1e-6);

        // Last inv_freq should be very small
        assert!(inv_freq_vec[63] < 0.01);

        Ok(())
    }

    #[test]
    fn test_cos_sin_shape() -> Result<()> {
        let rope = RotaryEmbedding::new(128, 1024, 1_000_000.0, DType::F32, &Device::Cpu)?;

        let (cos, sin) = rope.get_cos_sin(10, 0)?;
        // Shape is now [seq_len, head_dim/2] for fused kernel
        assert_eq!(cos.dims(), &[10, 64]);
        assert_eq!(sin.dims(), &[10, 64]);

        Ok(())
    }

    #[test]
    fn test_fused_rope_apply() -> Result<()> {
        let rope = RotaryEmbedding::new(128, 32, 1_000_000.0, DType::F32, &Device::Cpu)?;

        // Create dummy q and k tensors: [batch, heads, seq, head_dim]
        let q = Tensor::ones((1, 4, 8, 128), DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 4, 8, 128), DType::F32, &Device::Cpu)?;

        let (cos, sin) = rope.get_cos_sin(8, 0)?;
        let (q_rot, k_rot) = RotaryEmbedding::apply(&q, &k, &cos, &sin)?;

        // Output shapes should match input
        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());

        Ok(())
    }
}
