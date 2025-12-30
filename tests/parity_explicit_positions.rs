use std::collections::HashMap;
use std::fs;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use wedlm_rs::model::WeDLMForCausalLM;
use wedlm_rs::WeDLMConfig;

#[derive(Debug, Deserialize)]
struct TensorData {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct FixtureConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    hidden_act: String,
    max_position_embeddings: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    qk_norm: bool,
    attention_bias: bool,
    tie_word_embeddings: bool,
    eos_token_id: u32,
    pad_token_id: u32,
    mask_token_id: Option<u32>,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    config: FixtureConfig,
    input_ids: Vec<Vec<i64>>,
    positions: Vec<Vec<i64>>,
    weights: HashMap<String, TensorData>,
    expected_logits: TensorData,
}

fn build_config(cfg: &FixtureConfig) -> WeDLMConfig {
    WeDLMConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        intermediate_size: cfg.intermediate_size,
        hidden_act: cfg.hidden_act.clone(),
        max_position_embeddings: cfg.max_position_embeddings,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        tie_word_embeddings: cfg.tie_word_embeddings,
        eos_token_id: cfg.eos_token_id,
        pad_token_id: cfg.pad_token_id,
        mask_token_id: cfg.mask_token_id,
        dtype: cfg.dtype.clone(),
        rope_scaling: None,
        layer_types: vec!["full_attention".to_string(); cfg.num_hidden_layers],
    }
}

#[test]
fn test_explicit_positions_parity() -> Result<()> {
    let fixture_path = "tests/fixtures/wedlm_parity_small.json";
    let content = fs::read_to_string(fixture_path)
        .expect("Failed to read parity fixture JSON");
    let fixture: Fixture = serde_json::from_str(&content)
        .expect("Failed to parse parity fixture JSON");

    let config = build_config(&fixture.config);
    let device = Device::Cpu;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, td) in fixture.weights {
        let tensor = Tensor::from_vec(td.data, td.shape.as_slice(), &device)?;
        tensors.insert(name, tensor);
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let model = WeDLMForCausalLM::new(&config, vb, &device)?;

    let batch = fixture.input_ids.len();
    let seq_len = fixture.input_ids[0].len();
    let input_flat: Vec<i64> = fixture.input_ids.into_iter().flatten().collect();
    let pos_flat: Vec<i64> = fixture.positions.into_iter().flatten().collect();

    let input_ids = Tensor::from_vec(input_flat, (batch, seq_len), &device)?
        .to_dtype(DType::I64)?;
    let positions = Tensor::from_vec(pos_flat, (batch, seq_len), &device)?
        .to_dtype(DType::I64)?;

    let (logits, _) = model.forward_with_positions(&input_ids, &positions, None, None)?;
    let logits = logits.to_dtype(DType::F32)?;

    let logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let expected = fixture.expected_logits;

    let expected_len: usize = expected.shape.iter().product();
    assert_eq!(expected_len, expected.data.len(), "fixture expected_logits length mismatch");
    assert_eq!(logits_vec.len(), expected_len, "logits length mismatch");

    let mut max_diff = 0.0f32;
    for (a, b) in logits_vec.iter().zip(expected.data.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    let tol = 1e-3f32;
    assert!(
        max_diff < tol,
        "max diff {max_diff} exceeds tolerance {tol}"
    );

    Ok(())
}
