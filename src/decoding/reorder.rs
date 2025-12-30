//! Topological Reordering for WeDLM
//!
//! Moves non-MASK tokens to the front and MASK tokens to the end.
//! This allows MASK tokens to attend to the full prefix with causal attention.

use candle_core::{Device, Error, Result, Tensor};

/// Result of topological reordering (CPU-only, no tensors)
pub struct ReorderResult {
    /// Reordered token IDs as CPU vector
    pub reordered_ids: Vec<i64>,
    /// Mapping from reordered position to original position
    pub permutation: Vec<usize>,
    /// Inverse permutation (original -> reordered)
    pub inverse_perm: Vec<usize>,
    /// Number of non-MASK tokens
    pub num_known: usize,
}

/// Result of block reordering for WeDLM decode step
pub struct BlockReorderResult {
    /// Reordered block tokens (CPU vec, ready to upload)
    pub reordered_block: Vec<i64>,
    /// Position indices for reordered block (TRUE absolute positions)
    pub positions: Vec<i64>,
    /// Mapping from reordered block position to original block position
    pub block_permutation: Vec<usize>,
    /// Number of filled (non-MASK) tokens in block
    pub num_filled: usize,
}

/// Compute block reordering entirely on CPU - no GPU readback needed
///
/// # Arguments
/// * `block_tokens` - Current block tokens (CPU vec)
/// * `mask_positions` - Which positions in block are still MASKs (block-relative indices)
/// * `prefix_len` - Length of prefix (for computing absolute positions)
///
/// # Returns
/// Reordered block tokens and position indices, ready to upload to GPU
pub fn compute_block_reorder(
    block_tokens: &[i64],
    mask_positions: &[usize],
    prefix_len: usize,
) -> BlockReorderResult {
    let mut result = BlockReorderResult {
        reordered_block: Vec::with_capacity(block_tokens.len()),
        positions: Vec::with_capacity(block_tokens.len()),
        block_permutation: Vec::with_capacity(block_tokens.len()),
        num_filled: 0,
    };
    compute_block_reorder_into(block_tokens, mask_positions, prefix_len, &mut result);
    result
}

/// Compute block reordering into pre-allocated buffers (zero allocation in hot loop)
pub fn compute_block_reorder_into(
    block_tokens: &[i64],
    mask_positions: &[usize],
    prefix_len: usize,
    result: &mut BlockReorderResult,
) {
    let block_size = block_tokens.len();
    let mask_set: std::collections::HashSet<usize> = mask_positions.iter().copied().collect();

    // Clear and reuse buffers
    result.reordered_block.clear();
    result.positions.clear();
    result.block_permutation.clear();

    // Partition: filled positions first, then mask positions
    // First pass: add filled indices
    for i in 0..block_size {
        if !mask_set.contains(&i) {
            result.block_permutation.push(i);
        }
    }
    result.num_filled = result.block_permutation.len();

    // Second pass: add mask indices
    for i in 0..block_size {
        if mask_set.contains(&i) {
            result.block_permutation.push(i);
        }
    }

    // Build reordered tokens and positions from permutation
    for &block_pos in &result.block_permutation {
        result.reordered_block.push(block_tokens[block_pos]);
        result.positions.push((prefix_len + block_pos) as i64);
    }
}

/// Topological reordering: move non-MASK tokens to front, MASK tokens to end
/// Works entirely on CPU data - no GPU readback
///
/// # Arguments
/// * `ids` - Token IDs as CPU vector
/// * `mask_positions` - Explicit set of positions that are MASKs (absolute indices)
pub fn topological_reorder_cpu(
    ids: &[i64],
    mask_positions: &[usize],
) -> ReorderResult {
    let seq_len = ids.len();
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
    let reordered_ids: Vec<i64> = permutation.iter().map(|&i| ids[i]).collect();

    ReorderResult {
        reordered_ids,
        permutation,
        inverse_perm,
        num_known,
    }
}

/// Legacy function for compatibility - reads from GPU tensor
/// Prefer compute_block_reorder() or topological_reorder_cpu() to avoid GPU readback
pub fn topological_reorder(
    input_ids: &Tensor,
    mask_positions: &[usize],
) -> Result<ReorderResult> {
    let (batch_size, _seq_len) = input_ids.dims2()?;

    if batch_size != 1 {
        return Err(Error::Msg("topological_reorder only supports batch_size=1".to_string()));
    }

    // GPU READBACK - this is expensive on Metal, prefer CPU-only functions
    let ids: Vec<i64> = match input_ids.dtype() {
        candle_core::DType::I64 => input_ids.squeeze(0)?.to_vec1()?,
        _ => input_ids
            .squeeze(0)?
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as i64)
            .collect(),
    };

    Ok(topological_reorder_cpu(&ids, mask_positions))
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
    fn test_topological_reorder_cpu() {
        // [known, MASK, known, MASK] -> [known, known, MASK, MASK]
        let ids: Vec<i64> = vec![1, 151666, 2, 151666];

        // Explicitly specify mask positions (1 and 3)
        let mask_positions = vec![1, 3];
        let result = topological_reorder_cpu(&ids, &mask_positions);

        assert_eq!(result.num_known, 2);
        assert_eq!(result.permutation, vec![0, 2, 1, 3]);
        assert_eq!(result.reordered_ids, vec![1, 2, 151666, 151666]);
    }

    #[test]
    fn test_reorder_ignores_token_value() {
        // Token at position 1 happens to be 151666 (MASK ID) but is NOT a mask
        // Only position 3 is actually a mask
        let ids: Vec<i64> = vec![1, 151666, 2, 151666];

        // Only position 3 is a mask
        let mask_positions = vec![3];
        let result = topological_reorder_cpu(&ids, &mask_positions);

        // Position 1 should be treated as known (even though it has MASK token ID)
        assert_eq!(result.num_known, 3);
        assert_eq!(result.permutation, vec![0, 1, 2, 3]); // Only pos 3 moves to end
    }

    #[test]
    fn test_compute_block_reorder() {
        // Block with positions 0,1 filled, 2,3 are masks
        let block_tokens: Vec<i64> = vec![100, 200, 151666, 151666];
        let mask_positions = vec![2, 3]; // block-relative
        let prefix_len = 10;

        let result = compute_block_reorder(&block_tokens, &mask_positions, prefix_len);

        assert_eq!(result.num_filled, 2);
        assert_eq!(result.reordered_block, vec![100, 200, 151666, 151666]);
        // Positions are absolute: prefix_len + block_pos
        assert_eq!(result.positions, vec![10, 11, 12, 13]);
        assert_eq!(result.block_permutation, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_compute_block_reorder_mixed() {
        // Block: [MASK, filled, MASK, filled] -> [filled, filled, MASK, MASK]
        let block_tokens: Vec<i64> = vec![151666, 100, 151666, 200];
        let mask_positions = vec![0, 2]; // block-relative
        let prefix_len = 5;

        let result = compute_block_reorder(&block_tokens, &mask_positions, prefix_len);

        assert_eq!(result.num_filled, 2);
        assert_eq!(result.reordered_block, vec![100, 200, 151666, 151666]);
        // Original positions were 1,3,0,2 -> absolute: 6,8,5,7
        assert_eq!(result.positions, vec![6, 8, 5, 7]);
        assert_eq!(result.block_permutation, vec![1, 3, 0, 2]);
    }
}
