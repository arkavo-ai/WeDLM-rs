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
///
/// # Arguments
/// * `input_ids` - Token IDs [batch, seq_len]
/// * `mask_positions` - Explicit set of positions that are MASKs (absolute indices)
///
/// Using explicit mask positions avoids the issue where a predicted token
/// happens to equal the MASK token ID.
pub fn topological_reorder(
    input_ids: &Tensor,
    mask_positions: &[usize],
) -> Result<ReorderResult> {
    let (batch_size, seq_len) = input_ids.dims2()?;

    // For now, only support batch_size = 1
    if batch_size != 1 {
        return Err(Error::Msg("topological_reorder only supports batch_size=1".to_string()));
    }

    // Handle both I64 and U32 input dtypes
    let ids: Vec<u32> = match input_ids.dtype() {
        candle_core::DType::I64 => input_ids
            .squeeze(0)?
            .to_vec1::<i64>()?
            .iter()
            .map(|&x| x as u32)
            .collect(),
        _ => input_ids.squeeze(0)?.to_vec1()?,
    };
    let device = input_ids.device();

    // Convert mask_positions to a set for O(1) lookup
    let mask_set: std::collections::HashSet<usize> = mask_positions.iter().copied().collect();

    // Partition into known (non-MASK) and unknown (MASK) positions
    let mut known_indices: Vec<usize> = Vec::new();
    let mut unknown_indices: Vec<usize> = Vec::new();

    for i in 0..seq_len {
        if mask_set.contains(&i) {
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

    #[test]
    fn test_topological_reorder() -> Result<()> {
        let device = Device::Cpu;
        // [known, MASK, known, MASK] -> [known, known, MASK, MASK]
        let ids = Tensor::from_vec(vec![1u32, 151666, 2, 151666], (1, 4), &device)?;

        // Explicitly specify mask positions (1 and 3)
        let mask_positions = vec![1, 3];
        let result = topological_reorder(&ids, &mask_positions)?;

        assert_eq!(result.num_known, 2);
        assert_eq!(result.permutation, vec![0, 2, 1, 3]);

        let reordered: Vec<u32> = result.reordered_ids.squeeze(0)?.to_vec1()?;
        assert_eq!(reordered, vec![1, 2, 151666, 151666]);

        Ok(())
    }

    #[test]
    fn test_reorder_ignores_token_value() -> Result<()> {
        let device = Device::Cpu;
        // Token at position 1 happens to be 151666 (MASK ID) but is NOT a mask
        // Only position 3 is actually a mask
        let ids = Tensor::from_vec(vec![1u32, 151666, 2, 151666], (1, 4), &device)?;

        // Only position 3 is a mask
        let mask_positions = vec![3];
        let result = topological_reorder(&ids, &mask_positions)?;

        // Position 1 should be treated as known (even though it has MASK token ID)
        assert_eq!(result.num_known, 3);
        assert_eq!(result.permutation, vec![0, 1, 2, 3]); // Only pos 3 moves to end

        Ok(())
    }
}
