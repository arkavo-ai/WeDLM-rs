//! Topological Reordering for WeDLM
//!
//! Moves non-MASK tokens to the front and MASK tokens to the end.
//! This allows MASK tokens to attend to the full prefix with causal attention.

use candle_core::{Device, Error, Result, Tensor};

/// Result of topological reordering
pub struct ReorderResult {
    /// Reordered token IDs [batch, seq_len]
    pub reordered_ids: Tensor,
    /// Mapping from reordered position to original position
    pub permutation: Vec<usize>,
    /// Inverse permutation (original -> reordered)
    pub inverse_perm: Vec<usize>,
    /// Number of non-MASK tokens
    pub num_known: usize,
}

/// Topological reordering: move non-MASK tokens to front, MASK tokens to end
pub fn topological_reorder(
    input_ids: &Tensor,
    mask_token_id: u32,
) -> Result<ReorderResult> {
    let (batch_size, seq_len) = input_ids.dims2()?;

    // For now, only support batch_size = 1
    if batch_size != 1 {
        return Err(Error::Msg("topological_reorder only supports batch_size=1".to_string()));
    }

    let ids: Vec<u32> = input_ids.squeeze(0)?.to_vec1()?;
    let device = input_ids.device();

    // Partition into known (non-MASK) and unknown (MASK) positions
    let mut known_indices: Vec<usize> = Vec::new();
    let mut unknown_indices: Vec<usize> = Vec::new();

    for (i, &token_id) in ids.iter().enumerate() {
        if token_id == mask_token_id {
            unknown_indices.push(i);
        } else {
            known_indices.push(i);
        }
    }

    let num_known = known_indices.len();

    // Build permutation: known positions first, then unknown
    let mut permutation: Vec<usize> = Vec::with_capacity(seq_len);
    permutation.extend(&known_indices);
    permutation.extend(&unknown_indices);

    // Build inverse permutation
    let mut inverse_perm = vec![0usize; seq_len];
    for (new_pos, &old_pos) in permutation.iter().enumerate() {
        inverse_perm[old_pos] = new_pos;
    }

    // Reorder the token IDs
    let reordered: Vec<u32> = permutation.iter().map(|&i| ids[i]).collect();
    let reordered_ids =
        Tensor::from_vec(reordered, (1, seq_len), device)?.to_dtype(input_ids.dtype())?;

    Ok(ReorderResult {
        reordered_ids,
        permutation,
        inverse_perm,
        num_known,
    })
}

/// Reorder KV cache according to permutation
/// Uses index_select along the sequence dimension (dim 2)
/// Shape: [Batch, Heads, SeqLen, Dim] -> [Batch, Heads, NewSeqLen, Dim]
pub fn reorder_kv_cache(
    keys: &Tensor,
    values: &Tensor,
    permutation: &[usize],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Use u32 indices for Metal compatibility
    let perm_tensor = Tensor::from_vec(
        permutation.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        (permutation.len(),),
        device,
    )?;

    // index_select along sequence dimension (dim 2)
    let reordered_keys = keys.index_select(&perm_tensor, 2)?;
    let reordered_values = values.index_select(&perm_tensor, 2)?;

    Ok((reordered_keys, reordered_values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_topological_reorder() -> Result<()> {
        let device = Device::Cpu;
        // [known, MASK, known, MASK] -> [known, known, MASK, MASK]
        let ids = Tensor::from_vec(vec![1u32, 151666, 2, 151666], (1, 4), &device)?;

        let result = topological_reorder(&ids, 151666)?;

        assert_eq!(result.num_known, 2);
        assert_eq!(result.permutation, vec![0, 2, 1, 3]);

        let reordered: Vec<u32> = result.reordered_ids.squeeze(0)?.to_vec1()?;
        assert_eq!(reordered, vec![1, 2, 151666, 151666]);

        Ok(())
    }
}
