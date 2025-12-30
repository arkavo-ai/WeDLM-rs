//! WeDLM Block Decoding Algorithm
//!
//! Implements the core parallel decoding with topological reordering.

use candle_core::{Result, Tensor};

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
}

/// WeDLM Block Decoder
pub struct WeDLMDecoder<'a> {
    model: &'a WeDLMForCausalLM,
    mask_token_id: u32,
    eos_token_id: u32,
}

impl<'a> WeDLMDecoder<'a> {
    pub fn new(model: &'a WeDLMForCausalLM, mask_token_id: Option<u32>) -> Self {
        let config = model.config();
        Self {
            model,
            mask_token_id: mask_token_id.unwrap_or(MASK_TOKEN_ID),
            eos_token_id: config.eos_token_id,
        }
    }

    /// Generate one block of tokens using WeDLM parallel decoding
    pub fn generate_block(
        &self,
        prefix_ids: &Tensor,
        block_size: usize,
        params: &SamplingParams,
    ) -> Result<(Tensor, BlockStats)> {
        let device = prefix_ids.device();
        let dtype = prefix_ids.dtype();
        let prefix_len = prefix_ids.dim(1)?;
        let mut stats = BlockStats::default();

        // Step 1: Append MASK tokens
        let mask_block =
            Tensor::full(self.mask_token_id as i64, (1, block_size), device)?.to_dtype(dtype)?;
        let mut current_ids = Tensor::cat(&[prefix_ids, &mask_block], 1)?;
        let total_len = prefix_len + block_size;

        // Track which positions are still MASK
        let mut mask_positions: Vec<usize> = (prefix_len..total_len).collect();

        // Step 2: Iterate until all MASKs filled
        while !mask_positions.is_empty() {
            stats.steps += 1;

            // 2a: Topological reorder
            let reorder = topological_reorder(&current_ids, self.mask_token_id)?;

            // 2b: Forward pass (no KV cache for simplicity in block decoding)
            let (logits, _) = self.model.forward(&reorder.reordered_ids, 0, None)?;

            // 2c: Get logits only for MASK positions
            let num_mask = mask_positions.len();
            let mask_logits = logits.narrow(1, reorder.num_known, num_mask)?;
            let mask_logits = mask_logits.squeeze(0)?; // [num_mask, vocab_size]

            // 2d: Apply temperature and get predictions
            let (predictions, confidences) =
                sample_with_temperature(&mask_logits, params.temperature)?;

            // 2e: Select positions to fill
            let selected_indices = select_confident_positions(
                &confidences,
                params.confidence_threshold,
                params.max_tokens_per_step,
            )?;

            if selected_indices.is_empty() {
                break;
            }

            // 2f: Update sequence with predictions
            let pred_vec: Vec<i64> = predictions.to_vec1()?;
            let conf_vec: Vec<f32> = confidences.to_vec1()?;

            // Convert reordered MASK indices back to original positions
            let mut positions_to_fill: Vec<(usize, u32)> = Vec::new();
            for &reorder_idx in &selected_indices {
                let reordered_pos = reorder.num_known + reorder_idx;
                let original_pos = reorder.permutation[reordered_pos];
                let predicted_token = pred_vec[reorder_idx] as u32;

                positions_to_fill.push((original_pos, predicted_token));
                stats.avg_confidence += conf_vec[reorder_idx];
            }

            stats.tokens_generated += positions_to_fill.len();

            // Update the current_ids tensor
            let mut ids_vec: Vec<i64> = current_ids.squeeze(0)?.to_vec1()?;
            for (pos, token) in &positions_to_fill {
                ids_vec[*pos] = *token as i64;
            }
            current_ids = Tensor::from_vec(ids_vec, (1, total_len), device)?.to_dtype(dtype)?;

            // Update mask_positions
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

        // Return the full sequence (prefix + generated)
        Ok((current_ids, stats))
    }

    /// Generate multiple blocks until EOS or max_length
    pub fn generate(
        &self,
        prompt_ids: &Tensor,
        max_new_tokens: usize,
        block_size: usize,
        params: &SamplingParams,
    ) -> Result<Tensor> {
        let mut all_ids = prompt_ids.clone();
        let mut total_generated = 0;

        while total_generated < max_new_tokens {
            let remaining = max_new_tokens - total_generated;
            let current_block_size = remaining.min(block_size);

            let (new_ids, _stats) = self.generate_block(&all_ids, current_block_size, params)?;

            // Check for EOS in newly generated part
            let new_part_start = all_ids.dim(1)?;
            all_ids = new_ids;
            total_generated += current_block_size;

            let ids_vec: Vec<i64> = all_ids.squeeze(0)?.to_vec1()?;
            if ids_vec[new_part_start..]
                .iter()
                .any(|&t| t as u32 == self.eos_token_id)
            {
                break;
            }
        }

        Ok(all_ids)
    }
}
