//! High-level WeDLM inference engine

use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::config::WeDLMConfig;
use crate::decoding::{SamplingParams, WeDLMDecoder};
use crate::model::WeDLMForCausalLM;
use crate::weights::load_model_vb;
use crate::MASK_TOKEN_ID;

/// High-level inference engine
pub struct WeDLMEngine {
    model: WeDLMForCausalLM,
    tokenizer: Tokenizer,
    config: WeDLMConfig,
    device: Device,
}

impl WeDLMEngine {
    /// Load model and tokenizer from HuggingFace format directory
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Detect device
        let device = Self::select_device()?;
        tracing::info!("Using device: {:?}", device);

        // Load config
        let config = WeDLMConfig::from_file(model_path.join("config.json"))?;

        // Use F16 - Metal fully supports it including index_select for embeddings
        // BF16 has gaps in Metal support, F16 is the sweet spot for Apple Silicon
        let dtype = candle_core::DType::F16;
        tracing::info!(
            "Model config: {} layers, {} hidden, dtype={:?}",
            config.num_hidden_layers,
            config.hidden_size,
            dtype
        );

        // Load weights
        tracing::info!("Loading model weights...");
        let vb = load_model_vb(model_path, dtype, &device)?;

        // Build model
        tracing::info!("Building model...");
        let model = WeDLMForCausalLM::new(&config, vb, &device)?;

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
        })
    }

    /// Select the best available device
    fn select_device() -> Result<Device> {
        #[cfg(feature = "metal")]
        {
            if candle_core::utils::metal_is_available() {
                return Ok(Device::new_metal(0)?);
            }
        }

        #[cfg(feature = "cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                return Ok(Device::new_cuda(0)?);
            }
        }

        Ok(Device::Cpu)
    }

    /// Generate text using WeDLM block decoding
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        params: Option<SamplingParams>,
    ) -> Result<String> {
        self.generate_with_block_size(prompt, max_new_tokens, crate::DEFAULT_BLOCK_SIZE, params)
    }

    /// Generate text using WeDLM streaming parallel decoding
    ///
    /// Uses Algorithm 1 from the WeDLM paper with sliding window.
    pub fn generate_with_block_size(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        window_size: usize,
        params: Option<SamplingParams>,
    ) -> Result<String> {
        let params = params.unwrap_or_default();

        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();

        tracing::debug!("Prompt tokens: {}, window_size: {}", prompt_ids.len(), window_size);

        let prompt_tensor = candle_core::Tensor::from_vec(
            prompt_ids.clone(),
            (1, prompt_ids.len()),
            &self.device,
        )?;

        // Create decoder (mutable for prefix caching)
        let mut decoder = WeDLMDecoder::new(&self.model, Some(MASK_TOKEN_ID));

        // Generate using streaming (Algorithm 1)
        let (output_ids, _stats) = decoder.generate_streaming(&prompt_tensor, max_new_tokens, window_size, &params)?;

        // Decode output
        let output_vec: Vec<u32> = output_ids
            .squeeze(0)?
            .to_vec1::<i64>()?
            .iter()
            .map(|&x| x as u32)
            .collect();
        let text = self
            .tokenizer
            .decode(&output_vec, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(text)
    }

    /// Simple autoregressive generation (for comparison/debugging)
    pub fn generate_autoregressive(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();

        // Generate token by token
        for _ in 0..max_new_tokens {
            // Token IDs stay as I64 - embedding layer handles conversion
            let input = candle_core::Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;

            let (logits, _) = self.model.forward(&input, 0, None)?;

            // Get last token logits
            let seq_len = logits.dim(1)?;
            let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

            // Apply temperature and sample
            let scaled = if temperature > 0.0 {
                (last_logits / temperature as f64)?
            } else {
                last_logits
            };

            let probs = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;
            let next_token = probs.argmax(candle_core::D::Minus1)?.squeeze(0)?;
            let next_token: u32 = next_token.to_scalar()?;

            ids.push(next_token as i64);

            // Check for EOS
            if next_token == self.config.eos_token_id {
                break;
            }
        }

        // Decode
        let output_ids: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let text = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(text)
    }

    /// Get model configuration
    pub fn config(&self) -> &WeDLMConfig {
        &self.config
    }
}
