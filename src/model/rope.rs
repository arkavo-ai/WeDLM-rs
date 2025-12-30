//! Rotary Position Embeddings (RoPE) for WeDLM
//!
//! CRITICAL: This implementation must match the Python exactly.
//! - theta = 1,000,000.0 (WeDLM uses high theta for long context)
//! - head_dim = 128 -> 64 inverse frequencies
//! - Formula: inv_freq[i] = 1.0 / (theta ^ (2*i / head_dim))

use candle_core::{DType, Device, Result, Tensor, D};

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
        let (cos_cache, sin_cache) =
            Self::compute_cos_sin(&inv_freq, max_seq_len, head_dim, dtype, device)?;

        Ok(Self {
            inv_freq,
            head_dim,
            max_seq_len,
            cos_cache,
            sin_cache,
        })
    }

    /// Compute cosine and sine values for a range of positions
    fn compute_cos_sin(
        inv_freq: &Tensor,
        seq_len: usize,
        _head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // Position indices: [0, 1, 2, ..., seq_len-1]
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;

        // inv_freq shape: [half_dim] -> [1, half_dim]
        let inv_freq = inv_freq.unsqueeze(0)?;

        // freqs = positions @ inv_freq.T -> [seq_len, half_dim]
        // In Python: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        let freqs = positions.matmul(&inv_freq)?;

        // Concatenate to get full head_dim: [seq_len, head_dim]
        // emb = torch.cat((freqs, freqs), dim=-1)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        // Compute cos and sin
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

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

    /// Apply rotary embeddings to query and key tensors
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
    /// * `cos` - Cosine values [seq_len, head_dim]
    /// * `sin` - Sine values [seq_len, head_dim]
    ///
    /// # Returns
    /// (q_rotated, k_rotated) with same shapes as input
    pub fn apply(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Reshape cos/sin for broadcasting: [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let q_rotated = Self::apply_rotary_emb(q, &cos, &sin)?;
        let k_rotated = Self::apply_rotary_emb(k, &cos, &sin)?;

        Ok((q_rotated, k_rotated))
    }

    /// Apply rotary embedding to a single tensor
    fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // x shape: [batch, heads, seq, head_dim]
        // CRITICAL: Compute in f32 for numerical stability (matches Python)
        let input_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let cos_f32 = cos.to_dtype(DType::F32)?;
        let sin_f32 = sin.to_dtype(DType::F32)?;

        // q_embed = (q * cos) + (rotate_half(q) * sin)
        let x_rotated = rotate_half(&x_f32)?;
        let result = (x_f32.broadcast_mul(&cos_f32)? + x_rotated.broadcast_mul(&sin_f32)?)?;

        // Convert back to original dtype
        result.to_dtype(input_dtype)
    }
}

/// Rotate half the hidden dimensions
///
/// CRITICAL: Must match Python exactly:
/// ```python
/// x1 = x[..., : x.shape[-1] // 2]
/// x2 = x[..., x.shape[-1] // 2 :]
/// return torch.cat((-x2, x1), dim=-1)
/// ```
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let half = last_dim / 2;

    // Split into first and second halves
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;

    // Negate x2 and concatenate: (-x2, x1)
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], D::Minus1)
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
        assert_eq!(cos.dims(), &[10, 128]);
        assert_eq!(sin.dims(), &[10, 128]);

        Ok(())
    }

    #[test]
    fn test_rotate_half() -> Result<()> {
        // Test with simple values
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            (1, 1, 1, 4),
            &Device::Cpu,
        )?;

        let rotated = rotate_half(&x)?;
        let result: Vec<f32> = rotated.flatten_all()?.to_vec1()?;

        // (-x2, x1) = (-3, -4, 1, 2)
        assert_eq!(result, vec![-3.0, -4.0, 1.0, 2.0]);

        Ok(())
    }
}
