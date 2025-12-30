//! WeDLM Block Decoding Algorithm
//!
//! Implements parallel decoding with topological reordering.
//!
//! ## Key Insight: RoPE with Reordering
//!
//! WeDLM reorders tokens (non-masks first, masks last) for efficient parallel decoding.
//! However, RoPE must use the TRUE absolute positions, not the reordered sequence positions.
//!
//! Example:
//! - Original sequence: [A@0, B@1, MASK@2, C@3, MASK@4]
//! - Reordered tokens:  [A, B, C, MASK, MASK]
//! - RoPE positions:    [0, 1, 3, 2, 4]  <- TRUE positions, not [0,1,2,3,4]
//!
//! This allows caching prefix K/V while still doing parallel MASK prediction.

use candle_core::{DType, Result, Tensor};

use crate::model::WeDLMForCausalLM;
use crate::MASK_TOKEN_ID;

use super::reorder::topological_reorder;
use super::sampler::{sample_with_temperature, select_confident_positions, SamplingParams};

/// Statistics from block generation
#[derive(Debug, Default, Clone)]
pub struct BlockStats {
    pub steps: usize,
    pub tokens_generated: usize,
    pub avg_confidence: f32,
    /// Whether prefix cache was used
    pub prefix_cached: bool,
}

/// WeDLM Block Decoder with Stable Prefix Caching
pub struct WeDLMDecoder<'a> {
    model: &'a WeDLMForCausalLM,
    mask_token_id: u32,
    eos_token_id: u32,
    /// Cached K/V for stable prefix (one per layer)
    prefix_cache: Option<Vec<(Tensor, Tensor)>>,
    /// Length of cached prefix
    prefix_cache_len: usize,
}

impl<'a> WeDLMDecoder<'a> {
    pub fn new(model: &'a WeDLMForCausalLM, mask_token_id: Option<u32>) -> Self {
        let config = model.config();
        Self {
            model,
            mask_token_id: mask_token_id.unwrap_or(MASK_TOKEN_ID),
            eos_token_id: config.eos_token_id,
            prefix_cache: None,
            prefix_cache_len: 0,
        }
    }

    /// Compute and cache K/V for a prefix sequence
    fn cache_prefix(&mut self, prefix_ids: &Tensor) -> Result<()> {
        let prefix_len = prefix_ids.dim(1)?;

        // Forward pass for prefix only
        let (_, new_caches) = self.model.forward(prefix_ids, 0, None)?;

        self.prefix_cache = Some(new_caches);
        self.prefix_cache_len = prefix_len;

        Ok(())
    }

    /// Generate one block of tokens using WeDLM parallel decoding
    ///
    /// Uses topological reordering with correct RoPE positions:
    /// - Tokens are reordered (filled first, MASKs last) for efficient attention
    /// - RoPE uses TRUE absolute positions, preserving semantic correctness
    pub fn generate_block(
        &mut self,
        prefix_ids: &Tensor,
        block_size: usize,
        params: &SamplingParams,
    ) -> Result<(Tensor, BlockStats)> {
        let device = prefix_ids.device();
        let id_dtype = prefix_ids.dtype();  // I64 for token IDs
        let prefix_len = prefix_ids.dim(1)?;
        let mut stats = BlockStats::default();

        // Cache prefix if needed
        if self.prefix_cache.is_none() || self.prefix_cache_len != prefix_len {
            self.cache_prefix(prefix_ids)?;
        }
        stats.prefix_cached = true;

        // Initialize block tokens (all MASKs)
        let mut block_tokens: Vec<i64> = vec![self.mask_token_id as i64; block_size];

        // Track which positions in the block are still MASK
        let mut mask_positions: Vec<usize> = (0..block_size).collect();

        // Iterate until all MASKs filled
        while !mask_positions.is_empty() {
            stats.steps += 1;

            // Build current full sequence [prefix | block]
            let block_tensor = Tensor::from_vec(
                block_tokens.clone(),
                (1, block_size),
                device,
            )?.to_dtype(id_dtype)?;
            let current_ids = Tensor::cat(&[prefix_ids, &block_tensor], 1)?;

            // Convert block-relative mask_positions to absolute positions
            let absolute_mask_positions: Vec<usize> = mask_positions
                .iter()
                .map(|&block_pos| prefix_len + block_pos)
                .collect();

            // Topological reorder: known tokens first, MASKs last
            let reorder = topological_reorder(&current_ids, &absolute_mask_positions)?;

            // Build TRUE position indices for the reordered block tokens
            // Each reordered token maps back to its original absolute position
            let block_positions: Vec<i64> = (0..block_size)
                .map(|reorder_idx| {
                    // The token at reordered position (prefix_len + reorder_idx)
                    // came from original position permutation[prefix_len + reorder_idx]
                    let original_pos = reorder.permutation[prefix_len + reorder_idx];
                    original_pos as i64
                })
                .collect();

            let positions_tensor = Tensor::from_vec(
                block_positions,
                (1, block_size),
                device,
            )?;

            // Extract reordered block tokens
            let reordered_block = reorder.reordered_ids.narrow(1, prefix_len, block_size)?;

            // Compute how many tokens in block are filled vs MASK
            let num_filled_in_block = block_size - mask_positions.len();
            let num_mask = mask_positions.len();

            // Forward pass with:
            // - Reordered block tokens as input
            // - TRUE absolute positions for RoPE
            // - Cached prefix K/V
            // - Standard causal attention (Python uses FlashAttn with causal=True)
            let prefix_caches: Vec<Option<(Tensor, Tensor)>> = self
                .prefix_cache
                .as_ref()
                .unwrap()
                .iter()
                .map(|(k, v)| Some((k.clone(), v.clone())))
                .collect();

            let (logits, _block_caches) = self.model.forward_with_positions(
                &reordered_block,
                &positions_tensor,
                Some(&prefix_caches),
                None, // Use default causal attention mask
            )?;

            // Get logits for MASK positions (which are at the END in reordered sequence)
            let logits_2d = logits.squeeze(0)?;

            // MASKs are at positions num_filled_in_block..block_size in reordered output
            let mask_logits = logits_2d.narrow(0, num_filled_in_block, num_mask)?;

            // Apply temperature and get predictions
            let (predictions, confidences) =
                sample_with_temperature(&mask_logits, params.temperature)?;

            // Select positions to fill based on confidence
            let selected_indices = select_confident_positions(
                &confidences,
                params.confidence_threshold,
                params.max_tokens_per_step,
            )?;

            if selected_indices.is_empty() {
                break;
            }

            // Update block tokens with predictions
            let pred_vec: Vec<u32> = predictions.to_vec1()?;
            let conf_f32 = confidences.to_dtype(DType::F32)?;
            let conf_vec: Vec<f32> = conf_f32.to_vec1()?;

            let mut positions_to_fill: Vec<(usize, u32)> = Vec::new();
            for &reorder_mask_idx in &selected_indices {
                // Map from reordered MASK index to original block position
                let reordered_full_pos = prefix_len + num_filled_in_block + reorder_mask_idx;
                let original_full_pos = reorder.permutation[reordered_full_pos];
                let block_pos = original_full_pos - prefix_len;
                let predicted_token = pred_vec[reorder_mask_idx];

                positions_to_fill.push((block_pos, predicted_token));
                stats.avg_confidence += conf_vec[reorder_mask_idx];
            }

            stats.tokens_generated += positions_to_fill.len();

            // Update block_tokens
            for (pos, token) in &positions_to_fill {
                block_tokens[*pos] = *token as i64;
            }

            // Remove filled positions from mask tracking
            let filled_set: std::collections::HashSet<usize> =
                positions_to_fill.iter().map(|(p, _)| *p).collect();
            mask_positions.retain(|p| !filled_set.contains(p));

            // Check for EOS
            if positions_to_fill
                .iter()
                .any(|(_, t)| *t == self.eos_token_id)
            {
                break;
            }
        }

        if stats.tokens_generated > 0 {
            stats.avg_confidence /= stats.tokens_generated as f32;
        }

        // Build final output
        let final_block = Tensor::from_vec(block_tokens, (1, block_size), device)?.to_dtype(id_dtype)?;
        let output_ids = Tensor::cat(&[prefix_ids, &final_block], 1)?;

        Ok((output_ids, stats))
    }

    /// Commit completed block to prefix cache
    pub fn commit_block_to_cache(&mut self, full_sequence: &Tensor) -> Result<()> {
        let new_len = full_sequence.dim(1)?;
        let (_, new_caches) = self.model.forward(full_sequence, 0, None)?;
        self.prefix_cache = Some(new_caches);
        self.prefix_cache_len = new_len;
        Ok(())
    }

    /// Clear the prefix cache
    pub fn clear_cache(&mut self) {
        self.prefix_cache = None;
        self.prefix_cache_len = 0;
    }

    /// Generate multiple blocks until EOS or max_length
    pub fn generate(
        &mut self,
        prompt_ids: &Tensor,
        max_new_tokens: usize,
        block_size: usize,
        params: &SamplingParams,
    ) -> Result<Tensor> {
        let mut all_ids = prompt_ids.clone();
        let mut total_generated = 0;

        self.clear_cache();

        while total_generated < max_new_tokens {
            let remaining = max_new_tokens - total_generated;
            let current_block_size = remaining.min(block_size);

            let (new_ids, _stats) = self.generate_block(&all_ids, current_block_size, params)?;

            let new_part_start = all_ids.dim(1)?;
            all_ids = new_ids.clone();
            total_generated += current_block_size;

            let ids_vec: Vec<i64> = all_ids.squeeze(0)?.to_vec1()?;
            if ids_vec[new_part_start..]
                .iter()
                .any(|&t| t as u32 == self.eos_token_id)
            {
                break;
            }

            // Commit block to cache
            self.commit_block_to_cache(&all_ids)?;
        }

        Ok(all_ids)
    }
}
