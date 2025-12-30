//! Weight loading from sharded safetensors
//!
//! WeDLM-8B uses 4 safetensor shards:
//! - model-00001-of-00004.safetensors
//! - model-00002-of-00004.safetensors
//! - model-00003-of-00004.safetensors
//! - model-00004-of-00004.safetensors

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::Deserialize;

/// Index file structure for sharded models
#[derive(Debug, Deserialize)]
pub struct SafetensorsIndex {
    pub metadata: Option<serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

/// Load model from a directory containing config.json and safetensor shards
pub fn load_model_vb<P: AsRef<Path>>(
    model_dir: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let model_dir = model_dir.as_ref();
    let index_path = model_dir.join("model.safetensors.index.json");

    // Check if sharded or single file
    if index_path.exists() {
        load_sharded_vb(model_dir, dtype, device)
    } else {
        let single_path = model_dir.join("model.safetensors");
        if single_path.exists() {
            load_single_vb(&single_path, dtype, device)
        } else {
            anyhow::bail!("No model.safetensors or model.safetensors.index.json found")
        }
    }
}

/// Load from a single safetensors file
fn load_single_vb<P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let tensors = load_safetensors_file(path, dtype, device)?;
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

/// Load from sharded safetensors files
fn load_sharded_vb<P: AsRef<Path>>(
    model_dir: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let model_dir = model_dir.as_ref();
    let index_path = model_dir.join("model.safetensors.index.json");

    // Parse index file
    let index_content = std::fs::read_to_string(&index_path)
        .context("Failed to read model.safetensors.index.json")?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    // Group weights by shard file
    let mut shard_weights: HashMap<String, Vec<String>> = HashMap::new();
    for (weight_name, shard_file) in &index.weight_map {
        shard_weights
            .entry(shard_file.clone())
            .or_default()
            .push(weight_name.clone());
    }

    // Load each shard
    let mut all_tensors: HashMap<String, Tensor> = HashMap::new();
    let num_shards = shard_weights.len();
    tracing::info!("Loading {} weight shards...", num_shards);

    for (i, (shard_file, weight_names)) in shard_weights.iter().enumerate() {
        let shard_path = model_dir.join(shard_file);
        tracing::debug!(
            "Loading shard {}/{}: {} ({} tensors)",
            i + 1,
            num_shards,
            shard_file,
            weight_names.len()
        );

        let tensors = load_safetensors_file(&shard_path, dtype, device)?;

        for name in weight_names {
            if let Some(tensor) = tensors.get(name) {
                all_tensors.insert(name.clone(), tensor.clone());
            }
        }
    }

    tracing::info!("Loaded {} tensors total", all_tensors.len());
    Ok(VarBuilder::from_tensors(all_tensors, dtype, device))
}

/// Load tensors from a single safetensors file
fn load_safetensors_file<P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).context(format!("Failed to open {}", path.display()))?;

    // Memory-map for efficient loading
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut tensors = HashMap::new();

    for name in st.names() {
        let tensor_view = st.tensor(name)?;
        let tensor = tensor_from_view(&tensor_view, dtype, device)?;
        tensors.insert(name.to_string(), tensor);
    }

    Ok(tensors)
}

/// Convert safetensors TensorView to candle Tensor
fn tensor_from_view(
    view: &safetensors::tensor::TensorView,
    target_dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    use safetensors::Dtype as StDtype;

    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();

    let tensor = match view.dtype() {
        StDtype::BF16 => {
            let data: Vec<half::bf16> = data
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]))
                .collect();
            Tensor::from_vec(data, shape.as_slice(), device)?
        }
        StDtype::F16 => {
            let data: Vec<half::f16> = data
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                .collect();
            Tensor::from_vec(data, shape.as_slice(), device)?
        }
        StDtype::F32 => {
            let data: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::from_vec(data, shape.as_slice(), device)?
        }
        StDtype::F64 => {
            let data: Vec<f64> = data
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                })
                .collect();
            Tensor::from_vec(data, shape.as_slice(), device)?
        }
        StDtype::I32 => {
            let data: Vec<i32> = data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            // Convert to i64 for candle
            let data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
            Tensor::from_vec(data, shape.as_slice(), device)?
        }
        StDtype::I64 => {
            let data: Vec<i64> = data
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                })
                .collect();
            Tensor::from_vec(data, shape.as_slice(), device)?
        }
        other => anyhow::bail!("Unsupported dtype: {:?}", other),
    };

    // Convert to target dtype if needed and if it's a float type
    let tensor = if tensor.dtype() != target_dtype {
        match (tensor.dtype(), target_dtype) {
            (DType::BF16, DType::F32) | (DType::F16, DType::F32) | (DType::F32, DType::BF16) | (DType::F32, DType::F16) | (DType::BF16, DType::F16) | (DType::F16, DType::BF16) => {
                tensor.to_dtype(target_dtype)?
            }
            _ => tensor,
        }
    } else {
        tensor
    };

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_parse() {
        let json = r#"{
            "metadata": {"total_size": 16381470720},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                "lm_head.weight": "model-00004-of-00004.safetensors"
            }
        }"#;

        let index: SafetensorsIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.weight_map.len(), 2);
        assert_eq!(
            index.weight_map.get("model.embed_tokens.weight"),
            Some(&"model-00001-of-00004.safetensors".to_string())
        );
    }
}
