//! WeDLM Model Backbone
//!
//! The transformer backbone without the language modeling head.
//! Consists of embedding + layers + final norm.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::WeDLMConfig;

use super::attention::{create_causal_mask, RMSNorm};
use super::layer::WeDLMDecoderLayer;
use super::rope::RotaryEmbedding;

/// CPU-friendly embedding that works around Metal index_select limitations
/// Does lookup on CPU in F32, then converts and transfers to target device
struct CpuEmbedding {
    /// Embedding weights on CPU as F32 [vocab_size, hidden_size]
    weights: Tensor,
    /// Target device (Metal)
    target_device: Device,
    /// Target dtype (F16)
    target_dtype: DType,
}

impl CpuEmbedding {
    fn new(weights: Tensor, target_device: Device, target_dtype: DType) -> Result<Self> {
        // Move weights to CPU and convert to F32 for index_select compatibility
        let weights = weights.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        Ok(Self {
            weights,
            target_device,
            target_dtype,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Move input IDs to CPU and convert to U32 (required for index_select)
        let cpu_ids = input_ids.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;

        // Flatten for index_select
        let (batch_size, seq_len) = cpu_ids.dims2()?;
        let flat_ids = cpu_ids.flatten_all()?;

        // Do lookup on CPU with U32 indices
        let embeddings = self.weights.index_select(&flat_ids, 0)?;

        // Reshape back to [batch, seq, hidden]
        let hidden_size = self.weights.dim(1)?;
        let embeddings = embeddings.reshape((batch_size, seq_len, hidden_size))?;

        // Convert to target dtype and transfer to target device
        let embeddings = embeddings
            .to_dtype(self.target_dtype)?
            .to_device(&self.target_device)?;

        Ok(embeddings)
    }
}

/// WeDLM transformer backbone
pub struct WeDLMModel {
    /// Token embeddings (CPU-based for Metal compatibility)
    embed_tokens: CpuEmbedding,
    /// Decoder layers
    layers: Vec<WeDLMDecoderLayer>,
    /// Final layer norm
    norm: RMSNorm,
    /// Rotary position embeddings
    rope: RotaryEmbedding,
    /// Model configuration
    config: WeDLMConfig,
}

impl WeDLMModel {
    pub fn new(config: &WeDLMConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let dtype = config.candle_dtype();

        // Token embedding - load weights and wrap in CPU-friendly embedding
        let embed_weights = vb.pp("embed_tokens").get((config.vocab_size, config.hidden_size), "weight")?;
        let embed_tokens = CpuEmbedding::new(embed_weights, device.clone(), dtype)?;

        // Decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = WeDLMDecoderLayer::new(config, vb.pp(format!("layers.{}", i)))?;
            layers.push(layer);
        }

        // Final layer norm
        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        // Rotary embeddings
        let rope = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            config: config.clone(),
        })
    }

    /// Forward pass through the transformer
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `position_offset` - Starting position for RoPE (for incremental decoding)
    /// * `kv_caches` - Optional KV caches for each layer
    /// * `positions` - Optional explicit position indices [batch, seq_len] for WeDLM
    ///                 If None, uses sequential positions starting at position_offset
    ///
    /// # Returns
    /// (hidden_states, new_kv_caches)
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_offset: usize,
        kv_caches: Option<&[Option<(Tensor, Tensor)>]>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        self.forward_with_positions(input_ids, position_offset, kv_caches, None, None)
    }

    /// Forward pass with explicit position indices (for WeDLM parallel decoding)
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `position_offset` - Starting position for RoPE (ignored if positions provided)
    /// * `kv_caches` - Optional KV caches for each layer
    /// * `positions` - Optional explicit position indices [batch, seq_len] for WeDLM
    /// * `attention_mask` - Optional custom attention mask (if None, uses causal mask)
    pub fn forward_with_positions(
        &self,
        input_ids: &Tensor,
        position_offset: usize,
        kv_caches: Option<&[Option<(Tensor, Tensor)>]>,
        positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Get embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Get RoPE cos/sin - use explicit positions if provided
        let (cos, sin) = match positions {
            Some(pos) => self.rope.get_cos_sin_for_positions(pos)?,
            None => self.rope.get_cos_sin(seq_len, position_offset)?,
        };

        // Use provided attention mask or create default causal mask
        let attention_mask = match attention_mask {
            Some(mask) => mask.clone(),
            None => {
                let kv_len = match kv_caches {
                    Some(caches) => {
                        if let Some(Some((k, _))) = caches.first() {
                            k.dim(2)? + seq_len
                        } else {
                            seq_len
                        }
                    }
                    None => seq_len,
                };
                create_causal_mask(
                    seq_len,
                    kv_len,
                    self.config.candle_dtype(),
                    input_ids.device(),
                )?
            }
        };

        // Process through all layers
        let mut new_kv_caches = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let kv_cache = kv_caches.and_then(|c| c.get(i)).and_then(|c| c.as_ref());
            let kv_cache = kv_cache.map(|(k, v)| (k, v));

            let (new_hidden_states, new_kv) =
                layer.forward(&hidden_states, Some(&attention_mask), &cos, &sin, kv_cache)?;

            hidden_states = new_hidden_states;
            new_kv_caches.push(new_kv);
        }

        // Final layer norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        Ok((hidden_states, new_kv_caches))
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
