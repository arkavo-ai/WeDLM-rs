//! Sampling utilities for WeDLM decoding
//!
//! Implements the distance-penalized selection from the WeDLM paper:
//! "We define a distance-adjusted entropy: H̃_i = H_i + λ·d_i"
//! where d_i is the distance from slot i to the leftmost remaining mask.

use candle_core::{DType, Result, Tensor, D};

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

/// Sample from logits and compute per-position entropy and margin
///
/// # Returns
/// (predicted_tokens, entropies, margins) where:
/// - entropies[i] = H(p_i) (prediction entropy)
/// - margins[i] = logit(top1) - logit(top2) (confidence margin)
pub fn sample_with_entropy(
    logits: &Tensor,
    temperature: f32,
) -> Result<(Tensor, Vec<f32>, Vec<f32>)> {
    // Apply temperature
    let scaled_logits = if temperature != 1.0 && temperature > 0.0 {
        (logits / temperature as f64)?
    } else {
        logits.clone()
    };

    // Compute softmax probabilities
    let probs = candle_nn::ops::softmax(&scaled_logits, D::Minus1)?;

    // Get predictions (argmax)
    let predictions = probs.argmax(D::Minus1)?;

    // Compute entropy for each position: H = -sum(p * log(p))
    let probs_f32 = probs.to_dtype(DType::F32)?;
    let log_probs = (probs_f32.clone() + 1e-10)?.log()?;
    let neg_entropy = (probs_f32 * log_probs)?;
    let entropy_tensor = neg_entropy.sum(D::Minus1)?.neg()?;
    let entropies: Vec<f32> = entropy_tensor.to_vec1()?;

    // Compute margin for each position: logit(top1) - logit(top2)
    let logits_f32 = scaled_logits.to_dtype(DType::F32)?;
    let num_positions = logits_f32.dim(0)?;
    let mut margins = Vec::with_capacity(num_positions);

    for i in 0..num_positions {
        let pos_logits = logits_f32.get(i)?;
        let logits_vec: Vec<f32> = pos_logits.to_vec1()?;

        // Find top two logits
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for &l in &logits_vec {
            if l > top1 {
                top2 = top1;
                top1 = l;
            } else if l > top2 {
                top2 = l;
            }
        }
        margins.push(top1 - top2);
    }

    Ok((predictions, entropies, margins))
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
