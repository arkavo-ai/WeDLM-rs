//! WeDLM Decoder Layer
//!
//! Pre-norm architecture:
//! - residual + attention(norm(x))
//! - residual + mlp(norm(x))

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::WeDLMConfig;

use super::attention::{RMSNorm, WeDLMAttention};
use super::mlp::WeDLMMLP;

/// Single transformer decoder layer
pub struct WeDLMDecoderLayer {
    /// Self-attention
    self_attn: WeDLMAttention,
    /// Feed-forward MLP
    mlp: WeDLMMLP,
    /// Pre-attention layer norm
    input_layernorm: RMSNorm,
    /// Pre-MLP layer norm
    post_attention_layernorm: RMSNorm,
}

impl WeDLMDecoderLayer {
    pub fn new(config: &WeDLMConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = WeDLMAttention::new(config, vb.pp("self_attn"))?;
        let mlp = WeDLMMLP::new(config, vb.pp("mlp"))?;
        let input_layernorm =
            RMSNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RMSNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass through the decoder layer
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `attention_mask` - Optional mask [batch, 1, seq_len, kv_len]
    /// * `cos` - RoPE cosine values
    /// * `sin` - RoPE sine values
    /// * `kv_cache` - Optional KV cache from previous forward pass
    ///
    /// # Returns
    /// (output, new_kv_cache)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        // Pre-norm attention
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let (hidden_states, new_kv) =
            self.self_attn
                .forward(&hidden_states, attention_mask, cos, sin, kv_cache)?;
        let hidden_states = (residual + hidden_states)?;

        // Pre-norm MLP
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        Ok((hidden_states, new_kv))
    }
}
