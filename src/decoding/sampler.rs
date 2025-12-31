//! Sampling utilities for WeDLM decoding
//!
//! Implements the distance-penalized selection from the WeDLM paper:
//! "We define a distance-adjusted entropy: H̃_i = H_i + λ·d_i"
//! where d_i is the distance from slot i to the leftmost remaining mask.

use candle_core::{DType, Result, Tensor, D};
use rand::prelude::*;

/// Sampling parameters for WeDLM decoding
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Temperature for softmax (lower = more deterministic)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Entropy threshold τ for accepting predictions
    /// Lower = more conservative (higher quality, slower), higher = more aggressive (faster)
    pub entropy_threshold: f32,
    /// Distance penalty coefficient (unused with monotonic acceptance)
    pub distance_penalty: f32,
    /// Maximum tokens per step (limits parallelism)
    pub max_tokens_per_step: usize,
    /// Margin threshold: require logit(top1) - logit(top2) >= margin_threshold
    /// Higher = more conservative, set to 0 to disable
    pub margin_threshold: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.9,
            // Conservative default for quality - use higher for speed at quality cost
            entropy_threshold: 0.6,
            // Distance penalty λ for H̃ = H + λ·d selection (paper default)
            distance_penalty: 0.02,
            // Conservative parallelism - prevents lock-in under masked context
            max_tokens_per_step: 4,
            // Margin gate (disabled by default, streaming uses distance-penalized entropy)
            margin_threshold: 0.0,
        }
    }
}

/// Sample from logits with temperature
///
/// # Returns
/// (predicted_tokens, max_probabilities)
pub fn sample_with_temperature(
    logits: &Tensor,
    temperature: f32,
) -> Result<(Tensor, Tensor)> {
    // Apply temperature
    let scaled_logits = if temperature != 1.0 && temperature > 0.0 {
        (logits / temperature as f64)?
    } else {
        logits.clone()
    };

    // Compute softmax probabilities
    let probs = candle_nn::ops::softmax(&scaled_logits, D::Minus1)?;

    // Get max probability and predicted token for each position
    let max_probs = probs.max(D::Minus1)?;
    let predictions = probs.argmax(D::Minus1)?;

    Ok((predictions, max_probs))
}

/// Sample from logits with temperature and top-p, computing entropy and margin
///
/// Implements proper stochastic sampling per Algorithm 1's Sample(ℓ_i):
/// 1. Apply temperature scaling: ℓ' = ℓ / T
/// 2. Compute softmax probabilities
/// 3. Apply top-p (nucleus) filtering
/// 4. Sample from the filtered distribution
///
/// # Returns
/// (predicted_tokens, entropies, margins) where:
/// - predicted_tokens[i] = sampled token for position i
/// - entropies[i] = H(p_i) (prediction entropy, computed before top-p)
/// - margins[i] = logit(top1) - logit(top2) (confidence margin)
pub fn sample_with_entropy(
    logits: &Tensor,
    temperature: f32,
) -> Result<(Tensor, Vec<f32>, Vec<f32>)> {
    sample_with_entropy_and_top_p(logits, temperature, 1.0)
}

/// Sample from logits with temperature and top-p, computing entropy and margin
///
/// # Arguments
/// * `logits` - Shape [num_positions, vocab_size]
/// * `temperature` - Temperature for softmax scaling (0 = greedy/argmax)
/// * `top_p` - Nucleus sampling threshold (1.0 = no filtering)
pub fn sample_with_entropy_and_top_p(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
) -> Result<(Tensor, Vec<f32>, Vec<f32>)> {
    let (predictions, entropies, margins) = sample_impl(logits, temperature, top_p, true)?;
    Ok((predictions, entropies, margins.unwrap_or_default()))
}

/// Sample from logits without computing margins (faster for streaming)
///
/// Avoids per-position logit copy to CPU when margins aren't needed.
pub fn sample_without_margins(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
) -> Result<(Tensor, Vec<f32>)> {
    let (predictions, entropies, _) = sample_impl(logits, temperature, top_p, false)?;
    Ok((predictions, entropies))
}

/// Internal sampling implementation
///
/// Optimized for minimal GPU sync overhead: uses batch readback (single .to_vec2())
/// instead of per-position .get(i).to_vec1() calls. When `compute_margins` is false,
/// skips the logits readback entirely.
fn sample_impl(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    compute_margins: bool,
) -> Result<(Tensor, Vec<f32>, Option<Vec<f32>>)> {
    let num_positions = logits.dim(0)?;
    let device = logits.device();

    // Apply temperature scaling
    let scaled_logits = if temperature > 0.0 && temperature != 1.0 {
        (logits / temperature as f64)?
    } else {
        logits.clone()
    };

    // Compute softmax probabilities
    let probs = candle_nn::ops::softmax(&scaled_logits, D::Minus1)?;
    let probs_f32 = probs.to_dtype(DType::F32)?;

    // Compute entropy for each position: H = -sum(p * log(p))
    let log_probs = (probs_f32.clone() + 1e-10)?.log()?;
    let neg_entropy = (&probs_f32 * log_probs)?;
    let entropy_tensor = neg_entropy.sum(D::Minus1)?.neg()?;
    let entropies: Vec<f32> = entropy_tensor.to_vec1()?;

    // Batch readback: single GPU sync instead of N separate syncs per position
    let all_probs: Vec<Vec<f32>> = probs_f32.to_vec2()?;

    // Only readback logits if we need margins (single sync)
    let all_logits: Option<Vec<Vec<f32>>> = if compute_margins {
        Some(scaled_logits.to_dtype(DType::F32)?.to_vec2()?)
    } else {
        None
    };

    let mut margins = if compute_margins {
        Some(Vec::with_capacity(num_positions))
    } else {
        None
    };
    let mut sampled_tokens: Vec<u32> = Vec::with_capacity(num_positions);
    let mut rng = rand::thread_rng();

    for (i, pos_probs) in all_probs.into_iter().enumerate() {
        // Compute margin only if requested
        let top1_idx = if let (Some(ref logits_vec), Some(ref mut m)) = (&all_logits, &mut margins) {
            let pos_logits = &logits_vec[i];

            // Find top two logits for margin calculation
            let mut top1_logit = f32::NEG_INFINITY;
            let mut top2_logit = f32::NEG_INFINITY;
            let mut idx = 0usize;
            for (j, &l) in pos_logits.iter().enumerate() {
                if l > top1_logit {
                    top2_logit = top1_logit;
                    top1_logit = l;
                    idx = j;
                } else if l > top2_logit {
                    top2_logit = l;
                }
            }
            m.push(top1_logit - top2_logit);
            Some(idx)
        } else {
            None
        };

        // Sample token
        let token = if temperature == 0.0 {
            // Greedy: need argmax from probs
            if let Some(idx) = top1_idx {
                idx as u32
            } else {
                // Find argmax from probs
                pos_probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            }
        } else if top_p >= 1.0 {
            // No nucleus filtering: sample from full distribution
            sample_from_probs(&pos_probs, &mut rng)
        } else {
            // Nucleus (top-p) sampling
            sample_nucleus(&pos_probs, top_p, &mut rng)
        };
        sampled_tokens.push(token);
    }

    // Create predictions tensor (U32 dtype for token IDs)
    let predictions = Tensor::from_vec(sampled_tokens, (num_positions,), device)?;

    Ok((predictions, entropies, margins))
}

/// Sample a token index from a probability distribution
fn sample_from_probs<R: Rng>(probs: &[f32], rng: &mut R) -> u32 {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }
    // Fallback to last token (handles floating point rounding)
    (probs.len() - 1) as u32
}

/// Sample with nucleus (top-p) filtering
fn sample_nucleus<R: Rng>(probs: &[f32], top_p: f32, rng: &mut R) -> u32 {
    // Create (index, probability) pairs and sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find smallest set with cumulative probability >= top_p
    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, (_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize and sample from the nucleus
    let nucleus = &indexed[..cutoff_idx];
    let total: f32 = nucleus.iter().map(|(_, p)| p).sum();

    let r: f32 = rng.gen::<f32>() * total;
    let mut cumsum = 0.0;
    for (idx, p) in nucleus {
        cumsum += p;
        if r < cumsum {
            return *idx as u32;
        }
    }

    // Fallback to highest probability token
    indexed[0].0 as u32
}

/// Select positions to fill using monotonic (left-to-right) acceptance with dual gate
///
/// CRITICAL: Only accept a contiguous prefix of masked positions in original order.
/// This prevents "locking in" tokens predicted under partially-masked context.
///
/// Dual gate: position must pass BOTH:
/// 1. Entropy gate: entropy < entropy_threshold
/// 2. Margin gate: margin >= margin_threshold (if margin_threshold > 0)
///
/// # Arguments
/// * `entropies` - Per-position entropy values
/// * `margins` - Per-position margin values (logit_top1 - logit_top2)
/// * `mask_indices` - Original block positions of MASK tokens (in reordered order)
/// * `entropy_threshold` - τ threshold; positions with entropy < τ pass entropy gate
/// * `margin_threshold` - m threshold; positions with margin >= m pass margin gate
/// * `max_positions` - Maximum positions to accept
pub fn select_by_entropy_with_margin(
    entropies: &[f32],
    margins: &[f32],
    mask_indices: &[usize],
    entropy_threshold: f32,
    margin_threshold: f32,
    max_positions: usize,
) -> Vec<usize> {
    if entropies.is_empty() || mask_indices.is_empty() {
        return vec![];
    }

    // Create (reorder_idx, original_pos, entropy, margin) tuples
    let mut indexed: Vec<(usize, usize, f32, f32)> = entropies
        .iter()
        .zip(margins.iter())
        .enumerate()
        .map(|(i, (&h, &m))| {
            let original_pos = mask_indices.get(i).copied().unwrap_or(i);
            (i, original_pos, h, m)
        })
        .collect();

    // Sort by ORIGINAL position (left to right in output sequence)
    indexed.sort_by_key(|(_, orig_pos, _, _)| *orig_pos);

    // Accept contiguous prefix: stop at first position that fails EITHER gate
    let mut result: Vec<usize> = Vec::new();
    for (reorder_idx, _orig_pos, entropy, margin) in indexed {
        // Dual gate: must pass BOTH entropy AND margin
        let passes_entropy = entropy < entropy_threshold;
        let passes_margin = margin_threshold <= 0.0 || margin >= margin_threshold;

        if !passes_entropy || !passes_margin {
            // First failure - stop accepting
            break;
        }
        result.push(reorder_idx);
        if result.len() >= max_positions {
            break;
        }
    }

    // Fallback: if none selected, take the single leftmost position
    // (force progress even if uncertain)
    if result.is_empty() && !entropies.is_empty() {
        let mut indexed_for_fallback: Vec<(usize, usize)> = mask_indices
            .iter()
            .enumerate()
            .map(|(i, &orig)| (i, orig))
            .collect();
        indexed_for_fallback.sort_by_key(|(_, orig_pos)| *orig_pos);
        if let Some((leftmost_reorder_idx, _)) = indexed_for_fallback.first() {
            result.push(*leftmost_reorder_idx);
        }
    }

    result
}

/// Select positions using distance-penalized entropy from the WeDLM paper
///
/// Implements H̃_i = H_i + λ·d_i where:
/// - H_i = entropy of prediction at position i
/// - d_i = distance from position i to the leftmost remaining MASK (in logical window order)
/// - λ = distance_penalty coefficient
///
/// Selects ALL positions where H̃_i < τ (not monotonic - allows gaps).
/// Falls back to leftmost position if none selected.
///
/// # Arguments
/// * `entropies` - Per-position entropy values
/// * `mask_indices` - Logical window positions of MASK tokens
/// * `entropy_threshold` - τ threshold; positions with H̃ < τ are selected
/// * `distance_penalty` - λ coefficient for distance penalty
/// * `max_positions` - Maximum positions to accept
pub fn select_by_entropy_with_distance(
    entropies: &[f32],
    mask_indices: &[usize],
    entropy_threshold: f32,
    distance_penalty: f32,
    max_positions: usize,
) -> Vec<usize> {
    if entropies.is_empty() || mask_indices.is_empty() {
        return vec![];
    }

    // Find leftmost MASK position (minimum logical index)
    let leftmost_pos = *mask_indices.iter().min().unwrap_or(&0);

    // Calculate adjusted entropy H̃_i = H_i + λ·d_i for each position
    let mut indexed: Vec<(usize, usize, f32)> = entropies
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            let logical_pos = mask_indices.get(i).copied().unwrap_or(i);
            let distance = logical_pos.saturating_sub(leftmost_pos) as f32;
            let adjusted_entropy = h + distance_penalty * distance;
            (i, logical_pos, adjusted_entropy)
        })
        .collect();

    // Sort by logical position for consistent ordering
    indexed.sort_by_key(|(_, logical_pos, _)| *logical_pos);

    // Select positions where adjusted entropy < threshold
    let mut result: Vec<usize> = indexed
        .iter()
        .filter(|(_, _, adj_h)| *adj_h < entropy_threshold)
        .take(max_positions)
        .map(|(reorder_idx, _, _)| *reorder_idx)
        .collect();

    // Fallback: if none selected, take the single position with lowest adjusted entropy
    if result.is_empty() && !indexed.is_empty() {
        let min_idx = indexed
            .iter()
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _, _)| *idx)
            .unwrap();
        result.push(min_idx);
    }

    result
}

/// Legacy: Select positions based on confidence threshold
/// (Kept for backwards compatibility - prefer select_by_entropy_with_distance)
pub fn select_confident_positions(
    confidences: &Tensor,
    threshold: f32,
    max_positions: usize,
) -> Result<Vec<usize>> {
    let conf_f32 = confidences.to_dtype(DType::F32)?;
    let conf_vec: Vec<f32> = conf_f32.to_vec1()?;

    let mut candidates: Vec<(usize, f32)> = conf_vec
        .iter()
        .enumerate()
        .filter(|(_, &c)| c >= threshold)
        .map(|(i, &c)| (i, c))
        .collect();

    // Sort by confidence (descending)
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<usize> = candidates
        .into_iter()
        .take(max_positions)
        .map(|(i, _)| i)
        .collect();

    // Fallback to max confidence
    if selected.is_empty() && !conf_vec.is_empty() {
        let max_idx = conf_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();
        selected.push(max_idx);
    }

    Ok(selected)
}

/// Calculate entropy of a probability distribution tensor
pub fn calculate_entropy(probs: &Tensor) -> Result<f32> {
    let log_probs = (probs.clone() + 1e-10)?.log()?;
    let entropy = (probs * log_probs)?.sum_all()?.neg()?;
    Ok(entropy.to_scalar()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_select_by_entropy_with_distance() {
        // Test: positions 0,1,2,3 with entropies [0.3, 0.1, 0.5, 0.2]
        // mask_indices = [10, 11, 12, 13] (original block positions)
        // leftmost = 10, so distances = [0, 1, 2, 3]
        // With λ=0.1: adjusted = [0.3, 0.2, 0.7, 0.5]
        // With τ=0.4: selected indices where H̃ < 0.4 → [0, 1]
        
        let entropies = vec![0.3f32, 0.1, 0.5, 0.2];
        let mask_indices = vec![10, 11, 12, 13];
        
        let selected = select_by_entropy_with_distance(
            &entropies,
            &mask_indices,
            0.4,   // τ
            0.1,   // λ
            10,    // max
        );
        
        // Should select indices 0 and 1 (adjusted entropy 0.3 and 0.2)
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        assert!(!selected.contains(&2)); // 0.7 > 0.4
        assert!(!selected.contains(&3)); // 0.5 > 0.4
    }

    #[test]
    fn test_distance_penalty_biases_left() {
        // Same entropy everywhere, distance penalty should select leftmost
        let entropies = vec![0.3f32, 0.3, 0.3, 0.3];
        let mask_indices = vec![5, 6, 7, 8];
        
        let selected = select_by_entropy_with_distance(
            &entropies,
            &mask_indices,
            0.5,   // τ high enough to include some
            0.1,   // λ
            2,     // max_positions = 2
        );
        
        // Should prefer indices 0 and 1 (closer to leftmost)
        // Adjusted: [0.3, 0.4, 0.5, 0.6] - only [0,1] below threshold 0.5
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_fallback_to_min_entropy() {
        // All above threshold, should fallback to minimum adjusted entropy
        let entropies = vec![0.8f32, 0.9, 0.7];
        let mask_indices = vec![0, 1, 2];
        
        let selected = select_by_entropy_with_distance(
            &entropies,
            &mask_indices,
            0.3,   // τ very low
            0.0,   // λ = 0 (no distance penalty)
            10,
        );
        
        // Should fallback to index 2 (lowest entropy 0.7)
        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn test_select_confident() -> Result<()> {
        let conf = Tensor::from_vec(vec![0.5f32, 0.9, 0.3, 0.85], (4,), &Device::Cpu)?;
        let selected = select_confident_positions(&conf, 0.8, 10)?;
        assert!(selected.contains(&1));
        assert!(selected.contains(&3));
        assert_eq!(selected.len(), 2);
        Ok(())
    }

    #[test]
    fn test_fallback_to_max() -> Result<()> {
        let conf = Tensor::from_vec(vec![0.5f32, 0.6, 0.3], (3,), &Device::Cpu)?;
        let selected = select_confident_positions(&conf, 0.9, 10)?;
        assert_eq!(selected, vec![1]);
        Ok(())
    }
}
