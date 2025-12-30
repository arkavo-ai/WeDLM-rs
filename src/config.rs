//! WeDLM model configuration

use candle_core::DType;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for WeDLM-8B model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeDLMConfig {
    /// Vocabulary size: 151936 for WeDLM-8B
    pub vocab_size: usize,

    /// Hidden dimension: 4096
    pub hidden_size: usize,

    /// Number of transformer layers: 36
    pub num_hidden_layers: usize,

    /// Number of attention heads: 32
    pub num_attention_heads: usize,

    /// Number of key-value heads for GQA: 8
    pub num_key_value_heads: usize,

    /// Per-head dimension: 128
    pub head_dim: usize,

    /// MLP intermediate size: 12288
    pub intermediate_size: usize,

    /// Activation function: "silu"
    pub hidden_act: String,

    /// Maximum position embeddings: 16384
    pub max_position_embeddings: usize,

    /// RoPE theta: 1_000_000.0
    pub rope_theta: f64,

    /// RMSNorm epsilon: 1e-6
    pub rms_norm_eps: f64,

    /// Whether to use QK normalization: true for WeDLM
    #[serde(default)]
    pub qk_norm: bool,

    /// Whether attention has bias: false for WeDLM
    #[serde(default)]
    pub attention_bias: bool,

    /// Whether to tie word embeddings: false
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// EOS token ID: 151643
    #[serde(default)]
    pub eos_token_id: u32,

    /// PAD token ID: 151643
    #[serde(default)]
    pub pad_token_id: u32,

    /// MASK token ID for WeDLM decoding
    #[serde(default)]
    pub mask_token_id: Option<u32>,

    /// Data type: "bfloat16"
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Rope scaling configuration (null for WeDLM)
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    /// Layer types (all "full_attention" for WeDLM-8B)
    #[serde(default)]
    pub layer_types: Vec<String>,
}

fn default_dtype() -> String {
    "bfloat16".to_string()
}

impl WeDLMConfig {
    /// Load configuration from JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Compute the number of query heads per KV head group (for GQA)
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Get the candle DType based on config
    /// - If config says float32, always return F32 (for parity testing on CPU)
    /// - On Metal, return F16 (BF16 lacks full support)
    /// - Otherwise, parse the config dtype
    pub fn candle_dtype(&self) -> DType {
        // First parse what the config says
        let config_dtype = match self.dtype.to_lowercase().as_str() {
            "float32" | "f32" => DType::F32,
            "float16" | "f16" | "half" => DType::F16,
            "bfloat16" | "bf16" => DType::BF16,
            _ => DType::F32,
        };

        // If config explicitly says F32, honor it (for parity testing)
        if config_dtype == DType::F32 {
            return DType::F32;
        }

        // On Metal, use F16 for full compatibility (BF16 lacks index_select)
        #[cfg(feature = "metal")]
        {
            if candle_core::utils::metal_is_available() {
                return DType::F16;
            }
        }

        config_dtype
    }

    /// Create a default config matching WeDLM-8B-Instruct
    pub fn wedlm_8b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 16384,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            qk_norm: true,
            attention_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 151643,
            pad_token_id: 151643,
            mask_token_id: Some(151666),
            dtype: "bfloat16".to_string(),
            rope_scaling: None,
            layer_types: vec!["full_attention".to_string(); 36],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wedlm_8b_config() {
        let config = WeDLMConfig::wedlm_8b();
        assert_eq!(config.num_kv_groups(), 4); // 32 / 8 = 4
        assert_eq!(config.head_dim, 128);
        assert!(config.qk_norm);
        assert!(!config.attention_bias);
        assert_eq!(config.rope_theta, 1_000_000.0);
    }
}
