//! WeDLM for Causal Language Modeling
//!
//! Adds the language modeling head (lm_head) on top of the backbone.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::WeDLMConfig;

use super::backbone::WeDLMModel;

/// WeDLM for Causal Language Modeling
pub struct WeDLMForCausalLM {
    /// Transformer backbone
    model: WeDLMModel,
    /// Language modeling head: [hidden_size -> vocab_size]
    lm_head: Linear,
    /// Model configuration
    config: WeDLMConfig,
}

impl WeDLMForCausalLM {
    pub fn new(config: &WeDLMConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let model = WeDLMModel::new(config, vb.pp("model"), device)?;

        // LM head projects hidden states to vocabulary
        // Note: tie_word_embeddings is false for WeDLM, so separate weights
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            model,
            lm_head,
            config: config.clone(),
        })
    }

    /// Forward pass returning logits
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `position_offset` - Starting position for RoPE
    /// * `kv_caches` - Optional KV caches for each layer
    ///
    /// # Returns
    /// (logits, new_kv_caches) where logits is [batch, seq_len, vocab_size]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_offset: usize,
        kv_caches: Option<&[Option<(Tensor, Tensor)>]>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (hidden_states, new_kv_caches) =
            self.model.forward(input_ids, position_offset, kv_caches)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok((logits, new_kv_caches))
    }

    /// Forward pass with explicit position indices (for WeDLM parallel decoding)
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `positions` - Explicit position indices [batch, seq_len]
    /// * `kv_caches` - Optional KV caches for each layer
    /// * `attention_mask` - Optional custom attention mask
    ///
    /// # Returns
    /// (logits, new_kv_caches) where logits is [batch, seq_len, vocab_size]
    pub fn forward_with_positions(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&[Option<(Tensor, Tensor)>]>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (hidden_states, new_kv_caches) =
            self.model.forward_with_positions(input_ids, 0, kv_caches, Some(positions), attention_mask)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok((logits, new_kv_caches))
    }

    /// Get logits only for the last token (for efficient autoregressive generation)
    ///
    /// # Returns
    /// (logits, new_kv_caches) where logits is [batch, vocab_size]
    pub fn forward_last(
        &self,
        input_ids: &Tensor,
        position_offset: usize,
        kv_caches: Option<&[Option<(Tensor, Tensor)>]>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (logits, new_kv_caches) = self.forward(input_ids, position_offset, kv_caches)?;
        let seq_len = logits.dim(1)?;
        let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        Ok((last_logits, new_kv_caches))
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    /// Get model configuration
    pub fn config(&self) -> &WeDLMConfig {
        &self.config
    }
}
