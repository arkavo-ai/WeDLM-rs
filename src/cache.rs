//! KV Cache for efficient incremental decoding
//!
//! Stores past key-value pairs to avoid recomputation during autoregressive generation.

use candle_core::{Result, Tensor};

/// KV Cache for a single layer
#[derive(Clone)]
pub struct KVCache {
    /// Cached keys: [batch, num_kv_heads, cached_len, head_dim]
    keys: Option<Tensor>,
    /// Cached values: [batch, num_kv_heads, cached_len, head_dim]
    values: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
        }
    }

    /// Update cache with new keys and values
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.keys, &self.values) {
            (Some(cached_k), Some(cached_v)) => {
                let k = Tensor::cat(&[cached_k, new_k], 2)?;
                let v = Tensor::cat(&[cached_v, new_v], 2)?;
                (k, v)
            }
            _ => (new_k.clone(), new_v.clone()),
        };

        self.keys = Some(k.clone());
        self.values = Some(v.clone());

        Ok((k, v))
    }

    /// Get current cache length
    pub fn len(&self) -> usize {
        self.keys
            .as_ref()
            .map(|k| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
    }

    /// Get cached tensors as Option tuple
    pub fn get(&self) -> Option<(Tensor, Tensor)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Create KV caches for all layers
pub fn create_kv_caches(num_layers: usize) -> Vec<KVCache> {
    (0..num_layers).map(|_| KVCache::new()).collect()
}
