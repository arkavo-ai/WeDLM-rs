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

use candle_core::{Result, Tensor};

use crate::model::WeDLMForCausalLM;
use crate::MASK_TOKEN_ID;

use super::reorder::{compute_block_reorder_into, BlockReorderResult};
use super::sampler::{sample_with_entropy_and_top_p, sample_without_margins, select_by_entropy_with_distance, select_by_entropy_with_margin, SamplingParams};

/// Entropy threshold for reducing block size (soft limit)
pub const ENTROPY_SOFT_THRESHOLD: f32 = 8.0;

/// Entropy threshold for forcing single-token mode (hard limit)  
pub const ENTROPY_HARD_THRESHOLD: f32 = 12.0;

/// Maximum sequence length before mandatory cache refresh
pub const MAX_CACHED_LENGTH: usize = 16384;

/// Blocks between cache health checks
pub const CACHE_CHECK_INTERVAL: usize = 32;

/// Statistics from block generation
#[derive(Debug, Default, Clone)]
pub struct BlockStats {
    pub steps: usize,
    pub tokens_generated: usize,
    pub avg_confidence: f32,
    /// Average entropy of selected positions
    pub avg_entropy: f32,
    /// Maximum entropy seen in any step (for drift detection)
    pub max_entropy: f32,
    /// Number of high-entropy selections (above soft threshold)
    pub high_entropy_count: usize,
    /// Number of times block size was reduced due to entropy
    pub block_size_reductions: usize,
    /// Number of cache refreshes triggered
    pub cache_refreshes: usize,
    /// Whether prefix cache was used
    pub prefix_cached: bool,
}

/// WeDLM Block Decoder with Stable Prefix Caching
pub struct WeDLMDecoder<'a> {
    model: &'a WeDLMForCausalLM,
    mask_token_id: u32,
    eos_token_id: u32,
    /// Cached K/V for stable prefix (one per layer)
    /// Stored as Vec<Option<...>> to match forward_with_positions API and avoid per-step allocation
    prefix_cache: Vec<Option<(Tensor, Tensor)>>,
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
            prefix_cache: Vec::new(),
            prefix_cache_len: 0,
        }
    }

    /// Compute and cache K/V for a prefix sequence
    fn cache_prefix(&mut self, prefix_ids: &Tensor) -> Result<()> {
        let prefix_len = prefix_ids.dim(1)?;

        // Forward pass for prefix only
        let (_, new_caches) = self.model.forward(prefix_ids, 0, None)?;

        // Store as Vec<Option<...>> to match forward_with_positions API
        self.prefix_cache = new_caches.into_iter().map(Some).collect();
        self.prefix_cache_len = prefix_len;

        Ok(())
    }

    /// Generate one block of tokens using WeDLM parallel decoding
    ///
    /// Uses topological reordering with correct RoPE positions:
    /// - Tokens are reordered (filled first, MASKs last) for efficient attention
    /// - RoPE uses TRUE absolute positions, preserving semantic correctness
    ///
    /// # Returns
    /// (output_ids, block_tokens, stats) - block_tokens is the completed block on CPU
    pub fn generate_block(
        &mut self,
        prefix_ids: &Tensor,
        block_size: usize,
        params: &SamplingParams,
    ) -> Result<(Tensor, Vec<i64>, BlockStats)> {
        let device = prefix_ids.device();
        let prefix_len = prefix_ids.dim(1)?;
        let mut stats = BlockStats::default();

        // Cache prefix if needed
        if self.prefix_cache.is_empty() || self.prefix_cache_len != prefix_len {
            self.cache_prefix(prefix_ids)?;
        }
        stats.prefix_cached = true;

        // Initialize block tokens (all MASKs)
        let mut block_tokens: Vec<i64> = vec![self.mask_token_id as i64; block_size];

        // Track which positions in the block are still MASK
        let mut mask_positions: Vec<usize> = (0..block_size).collect();

        // Pre-allocate reusable buffers to avoid per-step allocations
        let mut reorder = BlockReorderResult {
            reordered_block: Vec::with_capacity(block_size),
            positions: Vec::with_capacity(block_size),
            block_permutation: Vec::with_capacity(block_size),
            num_filled: 0,
        };
        let mut positions_to_fill: Vec<(usize, u32)> = Vec::with_capacity(block_size);

        // Iterate until all MASKs filled
        while !mask_positions.is_empty() {
            stats.steps += 1;

            // Compute reordering entirely on CPU into pre-allocated buffers
            compute_block_reorder_into(&block_tokens, &mask_positions, prefix_len, &mut reorder);

            // Upload reordered block and positions to GPU (single upload per step)
            // Note: from_slice creates I64 tensor directly, no dtype conversion needed
            let reordered_block = Tensor::from_slice(
                &reorder.reordered_block,
                (1, block_size),
                device,
            )?;

            let positions_tensor = Tensor::from_slice(
                &reorder.positions,
                (1, block_size),
                device,
            )?;

            let num_filled_in_block = reorder.num_filled;
            let num_mask = mask_positions.len();

            // Forward pass with:
            // - Reordered block tokens as input
            // - TRUE absolute positions for RoPE
            // - Cached prefix K/V (passed as slice ref, no allocation)
            // - Standard causal attention (Python uses FlashAttn with causal=True)
            let (logits, _block_caches) = self.model.forward_with_positions(
                &reordered_block,
                &positions_tensor,
                Some(&self.prefix_cache),
                None, // Use default causal attention mask
            )?;

            // Get logits for MASK positions (which are at the END in reordered sequence)
            let logits_2d = logits.squeeze(0)?;

            // MASKs are at positions num_filled_in_block..block_size in reordered output
            let mask_logits = logits_2d.narrow(0, num_filled_in_block, num_mask)?;

            // Sample with entropy and margin calculation for dual-gate selection
            let (predictions, entropies, margins) =
                sample_with_entropy_and_top_p(&mask_logits, params.temperature, params.top_p)?;

            // Get original block positions for MASK tokens (for ordering)
            // MASKs are at the end of block_permutation after reordering
            let mask_original_positions: Vec<usize> = (num_filled_in_block..block_size)
                .map(|i| reorder.block_permutation[i])
                .collect();

            // Select positions using monotonic acceptance with dual gate:
            // 1. Entropy must be below threshold
            // 2. Margin (logit_top1 - logit_top2) must be above threshold
            let selected_indices = select_by_entropy_with_margin(
                &entropies,
                &margins,
                &mask_original_positions,
                params.entropy_threshold,
                params.margin_threshold,
                params.max_tokens_per_step,
            );

            if selected_indices.is_empty() {
                break;
            }

            // Update block tokens with predictions
            let pred_vec: Vec<u32> = predictions.to_vec1()?;

            positions_to_fill.clear();
            for &reorder_mask_idx in &selected_indices {
                // Map from reordered MASK index to original block position
                // MASKs are at the end: positions num_filled_in_block..block_size
                let reordered_block_pos = num_filled_in_block + reorder_mask_idx;
                let original_block_pos = reorder.block_permutation[reordered_block_pos];
                let predicted_token = pred_vec[reorder_mask_idx];

                positions_to_fill.push((original_block_pos, predicted_token));

                // Track entropy for drift detection
                let entropy = entropies[reorder_mask_idx];
                stats.avg_entropy += entropy;
                if entropy > stats.max_entropy {
                    stats.max_entropy = entropy;
                }
                if entropy > ENTROPY_SOFT_THRESHOLD {
                    stats.high_entropy_count += 1;
                }
            }

            stats.tokens_generated += positions_to_fill.len();

            // Update block_tokens
            for (pos, token) in &positions_to_fill {
                block_tokens[*pos] = *token as i64;
            }

            // Remove filled positions from mask tracking
            // Use linear scan instead of HashSet - faster for small N (max 32 elements)
            mask_positions.retain(|p| !positions_to_fill.iter().any(|(pos, _)| pos == p));

            // Check for EOS
            if positions_to_fill
                .iter()
                .any(|(_, t)| *t == self.eos_token_id)
            {
                break;
            }
        }

        if stats.tokens_generated > 0 {
            stats.avg_entropy /= stats.tokens_generated as f32;
            // Confidence ≈ 1 - normalized_entropy (rough approximation for compatibility)
            stats.avg_confidence = (1.0 - stats.avg_entropy / 10.0).max(0.0);
        }

        // Build final output
        let final_block = Tensor::from_vec(block_tokens.clone(), (1, block_size), device)?;
        let output_ids = Tensor::cat(&[prefix_ids, &final_block], 1)?;

        Ok((output_ids, block_tokens, stats))
    }

    /// Commit completed block to prefix cache incrementally
    ///
    /// Instead of recomputing the full sequence, we run just the block tokens
    /// with the existing prefix cache. The model concatenates K/V internally,
    /// so we get the combined (prefix + block) cache back.
    ///
    /// This is O(block_len) instead of O(prefix_len + block_len).
    pub fn commit_block_to_cache(&mut self, block_tokens: &[i64], device: &candle_core::Device) -> Result<()> {
        let block_size = block_tokens.len();

        // Build block tensor with sequential positions
        let block_tensor = Tensor::from_vec(
            block_tokens.to_vec(),
            (1, block_size),
            device,
        )?.to_dtype(candle_core::DType::I64)?;

        // Sequential positions starting after prefix
        let positions: Vec<i64> = (0..block_size)
            .map(|i| (self.prefix_cache_len + i) as i64)
            .collect();
        let positions_tensor = Tensor::from_vec(positions, (1, block_size), device)?;

        // Run just the block with prefix cache (passed as slice ref, no clone)
        let (_, new_caches) = self.model.forward_with_positions(
            &block_tensor,
            &positions_tensor,
            Some(&self.prefix_cache),
            None,
        )?;

        // Update cache in-place
        self.prefix_cache = new_caches.into_iter().map(Some).collect();
        self.prefix_cache_len += block_size;
        Ok(())
    }

    /// Clear the prefix cache
    pub fn clear_cache(&mut self) {
        self.prefix_cache.clear();
        self.prefix_cache_len = 0;
    }

    /// Truncate the KV cache to a specific length
    ///
    /// CRITICAL for streaming: After each forward pass, the cache includes
    /// K/V for window slots that will change. These must be discarded.
    fn truncate_cache(&mut self, new_len: usize) -> Result<()> {
        if new_len >= self.prefix_cache_len {
            return Ok(()); // Nothing to truncate
        }

        // Narrow each layer's K/V tensors along the sequence dimension (dim 2)
        // Shape: [batch, num_heads, seq_len, head_dim]
        for cache_opt in &mut self.prefix_cache {
            if let Some((k, v)) = cache_opt {
                *k = k.narrow(2, 0, new_len)?;
                *v = v.narrow(2, 0, new_len)?;
            }
        }
        self.prefix_cache_len = new_len;
        Ok(())
    }

    /// Streaming Parallel Decoding (Algorithm 1 from WeDLM paper)
    ///
    /// Key differences from block-based generation:
    /// 1. Commit-before-predict: tokens filled in step N are committed in step N+1
    /// 2. Cache truncation: discard window K/V after each forward (they become stale)
    /// 3. Sliding window: committed tokens leave, new MASKs enter
    /// 4. Continuous output: no block boundaries
    ///
    /// OPTIMIZATION: Only ONE forward pass per iteration by including committed
    /// tokens in the window. Their K/V is computed during the main forward pass,
    /// then we truncate to keep just committed K/V.
    pub fn generate_streaming(
        &mut self,
        prompt_ids: &Tensor,
        max_new_tokens: usize,
        window_size: usize,
        params: &SamplingParams,
    ) -> Result<(Tensor, BlockStats)> {
        let device = prompt_ids.device();
        let prefix_len = prompt_ids.dim(1)?;
        let mut stats = BlockStats::default();

        // 1. Cache the prompt
        self.clear_cache();
        self.cache_prefix(prompt_ids)?;
        stats.prefix_cached = true;

        // State tracking
        let mut committed_len = prefix_len;                    // Cache length (prompt + committed)
        let mut next_pos = prefix_len;                         // Next absolute position to assign
        let mut output_tokens: Vec<i64> = Vec::new();          // All committed output tokens

        // 2. Initialize window with MASKs at positions [prefix_len, prefix_len + window_size)
        // Clamp to max_new_tokens to avoid exceeding the limit
        let initial_window = window_size.min(max_new_tokens);
        let mut window_tokens: Vec<i64> = vec![self.mask_token_id as i64; initial_window];
        let mut window_positions: Vec<usize> = (next_pos..next_pos + initial_window).collect();
        next_pos += initial_window;

        // Pre-allocate reusable buffers to avoid per-iteration allocations
        let mut reorder = BlockReorderResult {
            reordered_block: Vec::with_capacity(window_size * 2),
            positions: Vec::with_capacity(window_size * 2),
            block_permutation: Vec::with_capacity(window_size * 2),
            num_filled: 0,
        };
        let mut mask_logical_indices: Vec<usize> = Vec::with_capacity(window_size);
        let mut reordered_pos: Vec<i64> = Vec::with_capacity(window_size);

        // Main generation loop
        while output_tokens.len() < max_new_tokens {
            stats.steps += 1;

            // ============================================================
            // STEP A: Count tokens to commit (contiguous non-MASK prefix)
            // These are from PREVIOUS iteration's predictions
            // ============================================================
            let n_commit = window_tokens
                .iter()
                .take_while(|&&t| t != self.mask_token_id as i64)
                .count();

            // ============================================================
            // STEP B: Build permutation (filled first, MASKs last)
            // Committed tokens are part of filled - their K/V will be
            // computed in this forward pass and kept in cache
            // ============================================================
            mask_logical_indices.clear();
            mask_logical_indices.extend(
                window_tokens
                    .iter()
                    .enumerate()
                    .filter(|(_, &t)| t == self.mask_token_id as i64)
                    .map(|(i, _)| i),
            );

            let current_window_size = window_tokens.len();

            // If window is empty or all filled, we need to commit and refill
            if current_window_size == 0 {
                break;
            }

            compute_block_reorder_into(
                &window_tokens,
                &mask_logical_indices,
                0, // We handle positions separately
                &mut reorder,
            );

            // Build reordered positions using window_positions
            reordered_pos.clear();
            reordered_pos.extend(
                reorder
                    .block_permutation
                    .iter()
                    .map(|&i| window_positions[i] as i64),
            );

            // Upload to GPU
            let reordered_tokens_tensor = Tensor::from_slice(
                &reorder.reordered_block,
                (1, current_window_size),
                device,
            )?;
            let positions_tensor = Tensor::from_slice(
                &reordered_pos,
                (1, current_window_size),
                device,
            )?;

            let num_filled = reorder.num_filled;
            let num_mask = mask_logical_indices.len();

            // ============================================================
            // STEP C: Forward pass with cache (single forward per iteration)
            // ============================================================
            let (logits, new_caches) = self.model.forward_with_positions(
                &reordered_tokens_tensor,
                &positions_tensor,
                Some(&self.prefix_cache),
                None,
            )?;

            // Store the updated cache (includes window K/V)
            self.prefix_cache = new_caches.into_iter().map(Some).collect();
            self.prefix_cache_len = committed_len + current_window_size;

            // ============================================================
            // STEP D: Cap n_commit to remaining capacity to avoid exceeding max_new_tokens
            // Then truncate cache to keep committed K/V, discard MASK K/V
            // ============================================================
            let remaining = max_new_tokens.saturating_sub(output_tokens.len());
            let n_commit = n_commit.min(remaining);
            let new_committed_len = committed_len + n_commit;
            self.truncate_cache(new_committed_len)?;

            // ============================================================
            // STEP D': Commit tokens - remove from window, add to output
            // ============================================================
            if n_commit > 0 {
                let committed_toks: Vec<i64> = window_tokens.drain(0..n_commit).collect();
                let _committed_pos: Vec<usize> = window_positions.drain(0..n_commit).collect();

                // Check for EOS
                let has_eos = committed_toks.iter().any(|&t| t as u32 == self.eos_token_id);

                // Add to output
                output_tokens.extend(&committed_toks);
                stats.tokens_generated += n_commit;
                committed_len = new_committed_len;

                if has_eos {
                    break;
                }

                // Refill window with new MASKs
                for _ in 0..n_commit {
                    if output_tokens.len() + window_tokens.len() >= max_new_tokens {
                        break;
                    }
                    window_tokens.push(self.mask_token_id as i64);
                    window_positions.push(next_pos);
                    next_pos += 1;
                }
            }

            // If no MASKs left to predict, continue to next iteration
            if num_mask == 0 {
                continue;
            }

            // ============================================================
            // STEP E: Predict + Fill using distance-penalized entropy
            // ============================================================
            let logits_2d = logits.squeeze(0)?;
            let mask_logits = logits_2d.narrow(0, num_filled, num_mask)?;

            let (predictions, entropies) =
                sample_without_margins(&mask_logits, params.temperature, params.top_p)?;

            // mask_logical_indices are BEFORE we drained n_commit tokens
            // We need to adjust indices for the positions AFTER commit
            // But actually, predictions are for the reordered MASK positions
            // which don't change - the reordering was done before commit
            let selected = select_by_entropy_with_distance(
                &entropies,
                &mask_logical_indices,
                params.entropy_threshold,
                params.distance_penalty,
                params.max_tokens_per_step,
            );

            // Fill selected positions
            // Note: mask_logical_indices are in pre-commit window space
            // After draining n_commit, indices shift by -n_commit
            let pred_vec: Vec<u32> = predictions.to_vec1()?;
            for &sel_idx in &selected {
                let pre_commit_pos = mask_logical_indices[sel_idx];
                // After draining n_commit, position shifts down
                let post_commit_pos = pre_commit_pos.saturating_sub(n_commit);

                if post_commit_pos < window_tokens.len() {
                    let predicted_token = pred_vec[sel_idx];
                    window_tokens[post_commit_pos] = predicted_token as i64;

                    // Track entropy stats
                    let entropy = entropies[sel_idx];
                    stats.avg_entropy += entropy;
                    if entropy > stats.max_entropy {
                        stats.max_entropy = entropy;
                    }
                    if entropy > ENTROPY_SOFT_THRESHOLD {
                        stats.high_entropy_count += 1;
                    }
                }
            }
        }

        // Finalize stats
        if stats.tokens_generated > 0 {
            stats.avg_entropy /= stats.tokens_generated as f32;
            stats.avg_confidence = (1.0 - stats.avg_entropy / 10.0).max(0.0);
        }

        // Build output tensor: prompt + all committed output
        let output_tensor = if output_tokens.is_empty() {
            prompt_ids.clone()
        } else {
            let output_part = Tensor::from_vec(output_tokens, (1, stats.tokens_generated), device)?;
            Tensor::cat(&[prompt_ids, &output_part], 1)?
        };

        Ok((output_tensor, stats))
    }

    /// Generate multiple blocks until EOS or max_length
    pub fn generate(
        &mut self,
        prompt_ids: &Tensor,
        max_new_tokens: usize,
        block_size: usize,
        params: &SamplingParams,
    ) -> Result<Tensor> {
        let (ids, _) = self.generate_with_stats(prompt_ids, max_new_tokens, block_size, params)?;
        Ok(ids)
    }

    /// Generate multiple blocks with cumulative statistics and production safeguards
    ///
    /// Includes:
    /// - Adaptive block sizing based on entropy
    /// - Automatic cache refresh for long sequences
    /// - Hard entropy limits that force conservative decoding
    pub fn generate_with_stats(
        &mut self,
        prompt_ids: &Tensor,
        max_new_tokens: usize,
        block_size: usize,
        params: &SamplingParams,
    ) -> Result<(Tensor, BlockStats)> {
        let device = prompt_ids.device();
        let mut all_ids = prompt_ids.clone();
        let mut total_generated = 0;
        let mut blocks_since_check = 0;
        
        // Adaptive block size - starts at requested, can shrink
        let mut effective_block_size = block_size;
        
        // Cumulative stats
        let mut cumulative = BlockStats::default();

        self.clear_cache();

        while total_generated < max_new_tokens {
            let remaining = max_new_tokens - total_generated;
            let current_block_size = remaining.min(effective_block_size);

            let (new_ids, block_tokens, block_stats) = 
                self.generate_block(&all_ids, current_block_size, params)?;

            // Accumulate stats
            cumulative.steps += block_stats.steps;
            cumulative.tokens_generated += block_stats.tokens_generated;
            cumulative.high_entropy_count += block_stats.high_entropy_count;
            if block_stats.max_entropy > cumulative.max_entropy {
                cumulative.max_entropy = block_stats.max_entropy;
            }
            
            // Running average entropy
            if cumulative.tokens_generated > 0 {
                let prev_tokens = cumulative.tokens_generated - block_stats.tokens_generated;
                let prev_weight = prev_tokens as f32;
                let new_weight = block_stats.tokens_generated as f32;
                cumulative.avg_entropy = (cumulative.avg_entropy * prev_weight 
                    + block_stats.avg_entropy * new_weight) 
                    / cumulative.tokens_generated as f32;
            }

            // PRODUCTION SAFEGUARD 1: Adaptive block sizing
            if block_stats.max_entropy > ENTROPY_HARD_THRESHOLD {
                // Force single-token mode - model is too uncertain
                effective_block_size = 1;
                cumulative.block_size_reductions += 1;
            } else if block_stats.max_entropy > ENTROPY_SOFT_THRESHOLD {
                // Reduce block size by half
                effective_block_size = (effective_block_size / 2).max(1);
                cumulative.block_size_reductions += 1;
            } else if effective_block_size < block_size && block_stats.avg_entropy < ENTROPY_SOFT_THRESHOLD / 2.0 {
                // Entropy recovered - gradually restore block size
                effective_block_size = (effective_block_size * 2).min(block_size);
            }

            all_ids = new_ids;
            total_generated += current_block_size;
            blocks_since_check += 1;

            // Check for EOS
            if block_tokens.iter().any(|&t| t as u32 == self.eos_token_id) {
                break;
            }

            // Commit block to cache
            self.commit_block_to_cache(&block_tokens, device)?;

            // PRODUCTION SAFEGUARD 2: Cache length limit
            if self.prefix_cache_len > MAX_CACHED_LENGTH {
                // Refresh cache from current sequence
                self.refresh_cache_from_sequence(&all_ids)?;
                cumulative.cache_refreshes += 1;
            }

            // PRODUCTION SAFEGUARD 3: Periodic cache health check
            if blocks_since_check >= CACHE_CHECK_INTERVAL {
                blocks_since_check = 0;
                
                // If entropy has been consistently high, refresh cache
                if cumulative.high_entropy_count > CACHE_CHECK_INTERVAL / 2 {
                    self.refresh_cache_from_sequence(&all_ids)?;
                    cumulative.cache_refreshes += 1;
                    cumulative.high_entropy_count = 0; // Reset after refresh
                }
            }
        }
        
        cumulative.avg_confidence = (1.0 - cumulative.avg_entropy / 10.0).max(0.0);
        cumulative.prefix_cached = true;

        Ok((all_ids, cumulative))
    }

    /// Refresh the KV cache by recomputing from the full sequence
    /// 
    /// This fixes accumulated numerical errors in long-context generation.
    fn refresh_cache_from_sequence(&mut self, sequence: &Tensor) -> Result<()> {
        self.clear_cache();
        self.cache_prefix(sequence)?;
        Ok(())
    }
}
