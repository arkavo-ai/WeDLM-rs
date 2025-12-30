//! Sampling utilities for WeDLM decoding

use candle_core::{Result, Tensor, D};

/// Sampling parameters for WeDLM decoding
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Temperature for softmax (lower = more deterministic)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Confidence threshold for accepting predictions
    pub confidence_threshold: f32,
    /// Maximum tokens per step (limits parallelism)
    pub max_tokens_per_step: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.9,
            confidence_threshold: 0.8,
            max_tokens_per_step: 8,
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

/// Select positions to fill based on confidence
pub fn select_confident_positions(
    confidences: &Tensor,
    threshold: f32,
    max_positions: usize,
) -> Result<Vec<usize>> {
    // Convert to F32 if needed (model outputs F16 on Metal)
    let conf_f32 = confidences.to_dtype(candle_core::DType::F32)?;
    let conf_vec: Vec<f32> = conf_f32.to_vec1()?;

    // Find positions above threshold
    let mut candidates: Vec<(usize, f32)> = conf_vec
        .iter()
        .enumerate()
        .filter(|(_, &c)| c >= threshold)
        .map(|(i, &c)| (i, c))
        .collect();

    // Sort by confidence (descending)
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top max_positions
    let mut selected: Vec<usize> = candidates
        .into_iter()
        .take(max_positions)
        .map(|(i, _)| i)
        .collect();

    // If none above threshold, take the single most confident
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

/// Calculate entropy of a probability distribution
pub fn calculate_entropy(probs: &Tensor) -> Result<f32> {
    // H = -sum(p * log(p))
    let log_probs = probs.log()?;
    let entropy = (probs * log_probs)?.sum_all()?.neg()?;
    let entropy: f32 = entropy.to_scalar()?;
    Ok(entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_select_confident() -> Result<()> {
        let conf = Tensor::from_vec(vec![0.5f32, 0.9, 0.3, 0.85], (4,), &Device::Cpu)?;

        // Threshold 0.8 should select indices 1 and 3
        let selected = select_confident_positions(&conf, 0.8, 10)?;
        assert!(selected.contains(&1));
        assert!(selected.contains(&3));
        assert_eq!(selected.len(), 2);

        Ok(())
    }

    #[test]
    fn test_fallback_to_max() -> Result<()> {
        let conf = Tensor::from_vec(vec![0.5f32, 0.6, 0.3], (3,), &Device::Cpu)?;

        // Threshold 0.9 should fallback to index 1 (max confidence)
        let selected = select_confident_positions(&conf, 0.9, 10)?;
        assert_eq!(selected, vec![1]);

        Ok(())
    }
}
