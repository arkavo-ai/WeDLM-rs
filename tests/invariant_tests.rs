//! Invariant tests to localize WeDLM bug
//!
//! These tests verify fundamental correctness properties that MUST hold
//! for WeDLM to work correctly. They help identify whether bugs are in:
//! - (a) reorder/mapping/position shift logic, OR
//! - (b) sampler/entropy selection logic
//!
//! Run with: `cargo test invariant --release -- --ignored`

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::ops::softmax;
use tokenizers::Tokenizer;

use wedlm_rs::config::WeDLMConfig;
use wedlm_rs::decoding::{WeDLMDecoder, SamplingParams};
use wedlm_rs::model::WeDLMForCausalLM;
use wedlm_rs::weights::load_model_vb;
use wedlm_rs::MASK_TOKEN_ID;

/// Helper to get model path from environment or default HF cache
fn get_model_path() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("WEDLM_MODEL_PATH") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    // Try default HuggingFace cache location
    let home = std::env::var("HOME").ok()?;
    let hf_path = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--tencent--WeDLM-8B-Instruct/snapshots");

    if hf_path.exists() {
        // Find latest snapshot
        if let Ok(entries) = std::fs::read_dir(&hf_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && path.join("config.json").exists() {
                    return Some(path);
                }
            }
        }
    }

    None
}

/// Load model and tokenizer for testing
fn load_test_model() -> Result<(WeDLMForCausalLM, Tokenizer, WeDLMConfig, Device)> {
    let model_path = get_model_path()
        .expect("Model not found. Set WEDLM_MODEL_PATH or download tencent/WeDLM-8B-Instruct");

    let device = {
        #[cfg(feature = "metal")]
        {
            if candle_core::utils::metal_is_available() {
                Device::new_metal(0)?
            } else {
                Device::Cpu
            }
        }
        #[cfg(not(feature = "metal"))]
        Device::Cpu
    };

    let config = WeDLMConfig::from_file(model_path.join("config.json"))
        .map_err(|e| candle_core::Error::Msg(format!("Config load failed: {}", e)))?;

    let dtype = DType::F16;
    let vb = load_model_vb(&model_path, dtype, &device)
        .map_err(|e| candle_core::Error::Msg(format!("Model load failed: {}", e)))?;
    let model = WeDLMForCausalLM::new(&config, vb, &device)?;

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| candle_core::Error::Msg(format!("Tokenizer load failed: {}", e)))?;

    Ok((model, tokenizer, config, device))
}

/// Invariant A: With no masks, WeDLM forward should match AR forward
///
/// This tests that when we pass a fully-filled block (no MASKs) through
/// forward_with_positions, the logits match what we get from a standard
/// forward pass.
///
/// If this FAILS: The reorder path (position_ids, attention_mask, or token
/// ordering) is broken even before masking enters the picture.
#[test]
#[ignore] // Requires model download - run with `cargo test -- --ignored`
fn test_invariant_a_no_masks_matches_ar() -> Result<()> {
    println!("\n=== Invariant A: No-masks should match AR forward ===\n");

    let (model, tokenizer, _config, device) = load_test_model()?;

    // Create a prompt
    let prompt = "The quick brown fox";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenize failed: {}", e)))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let prompt_len = prompt_ids.len();

    println!("Prompt: {:?}", prompt);
    println!("Token IDs: {:?}", prompt_ids);
    println!("Prompt length: {}", prompt_len);

    // Create the full sequence tensor
    let input_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), &device)?;

    // === AR Forward Pass ===
    // Standard forward: model.forward(input_ids, position_offset=0, cache=None)
    let (ar_logits, _) = model.forward(&input_tensor, 0, None)?;
    let ar_logits = ar_logits.to_dtype(DType::F32)?;

    // Get logits for last position (what AR would use for next token prediction)
    let ar_last_logits = ar_logits.narrow(1, prompt_len - 1, 1)?.squeeze(1)?;
    println!("AR logits shape: {:?}", ar_logits.dims());

    // === WeDLM-style Forward Pass ===
    // Use forward_with_positions with explicit positions [0, 1, 2, ..., N-1]
    let positions: Vec<i64> = (0..prompt_len as i64).collect();
    let positions_tensor = Tensor::from_vec(positions, (1, prompt_len), &device)?;

    let (wedlm_logits, _) = model.forward_with_positions(
        &input_tensor,
        &positions_tensor,
        None, // No cache
        None, // No custom attention mask
    )?;
    let wedlm_logits = wedlm_logits.to_dtype(DType::F32)?;

    // Get logits for last position
    let wedlm_last_logits = wedlm_logits.narrow(1, prompt_len - 1, 1)?.squeeze(1)?;
    println!("WeDLM logits shape: {:?}", wedlm_logits.dims());

    // === Compare ===
    let ar_vec: Vec<f32> = ar_last_logits.flatten_all()?.to_vec1()?;
    let wedlm_vec: Vec<f32> = wedlm_last_logits.flatten_all()?.to_vec1()?;

    assert_eq!(ar_vec.len(), wedlm_vec.len(), "Logits length mismatch");

    let mut max_diff = 0.0f32;
    let mut diff_count = 0;
    for (i, (a, w)) in ar_vec.iter().zip(wedlm_vec.iter()).enumerate() {
        let diff = (a - w).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-3 && diff_count < 5 {
            println!("Diff at {}: AR={:.6}, WeDLM={:.6}, diff={:.6}", i, a, w, diff);
            diff_count += 1;
        }
    }

    println!("\nMax difference: {:.6}", max_diff);

    // Get predictions
    let ar_pred: u32 = ar_last_logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;
    let wedlm_pred: u32 = wedlm_last_logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("AR prediction: {} -> {:?}", ar_pred, tokenizer.decode(&[ar_pred], false));
    println!("WeDLM prediction: {} -> {:?}", wedlm_pred, tokenizer.decode(&[wedlm_pred], false));

    // Tolerance: F16 computations can have small differences
    let tol = 1e-2f32;
    assert!(
        max_diff < tol,
        "INVARIANT A FAILED: max diff {:.6} exceeds tolerance {:.6}\n\
         This indicates the forward_with_positions path differs from standard forward.\n\
         Bug is likely in position handling or attention computation.",
        max_diff, tol
    );

    assert_eq!(
        ar_pred, wedlm_pred,
        "INVARIANT A FAILED: Predictions differ! AR={}, WeDLM={}\n\
         Even if logits are close, predictions should match.",
        ar_pred, wedlm_pred
    );

    println!("\n✓ Invariant A PASSED: No-masks forward matches AR\n");

    Ok(())
}

/// Invariant B: Single-mask prediction should match AR next-token
///
/// This tests that when we predict a SINGLE masked position using WeDLM,
/// the prediction matches what autoregressive would generate at that position.
///
/// If this FAILS: The "which logits row predicts which position" mapping
/// is wrong - likely an off-by-one error in position indexing.
#[test]
#[ignore] // Requires model download - run with `cargo test -- --ignored`
fn test_invariant_b_single_mask_matches_ar() -> Result<()> {
    println!("\n=== Invariant B: Single-mask should match AR prediction ===\n");

    let (model, tokenizer, _config, device) = load_test_model()?;

    // Create a prompt
    let prompt = "The quick brown";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenize failed: {}", e)))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let prompt_len = prompt_ids.len();

    println!("Prompt: {:?}", prompt);
    println!("Token IDs: {:?}", prompt_ids);
    println!("Prompt length: {}", prompt_len);

    // === AR: Get next token prediction ===
    let input_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), &device)?;
    let (ar_logits, _) = model.forward(&input_tensor, 0, None)?;
    let ar_logits = ar_logits.to_dtype(DType::F32)?;

    // Get logits for position after prompt (what we're predicting)
    let ar_last_logits = ar_logits.narrow(1, prompt_len - 1, 1)?.squeeze(1)?;
    let ar_probs = softmax(&ar_last_logits, D::Minus1)?;
    let ar_pred: u32 = ar_probs.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("AR next token prediction: {} -> {:?}", ar_pred, tokenizer.decode(&[ar_pred], false));

    // === WeDLM: Predict single MASK at position prompt_len ===
    // Sequence: [prompt_tokens..., MASK]
    // Positions: [0, 1, ..., prompt_len-1, prompt_len]
    let mut wedlm_ids = prompt_ids.clone();
    wedlm_ids.push(MASK_TOKEN_ID as i64);
    let wedlm_len = wedlm_ids.len();

    println!("WeDLM input: {:?}", wedlm_ids);
    println!("MASK token ID: {}", MASK_TOKEN_ID);

    let wedlm_input = Tensor::from_vec(wedlm_ids, (1, wedlm_len), &device)?;
    let positions: Vec<i64> = (0..wedlm_len as i64).collect();
    let positions_tensor = Tensor::from_vec(positions, (1, wedlm_len), &device)?;

    let (wedlm_logits, _) = model.forward_with_positions(
        &wedlm_input,
        &positions_tensor,
        None,
        None,
    )?;
    let wedlm_logits = wedlm_logits.to_dtype(DType::F32)?;

    // Get logits for the MASK position (last position in sequence)
    let wedlm_mask_logits = wedlm_logits.narrow(1, wedlm_len - 1, 1)?.squeeze(1)?;
    let wedlm_probs = softmax(&wedlm_mask_logits, D::Minus1)?;
    let wedlm_pred: u32 = wedlm_probs.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("WeDLM MASK prediction: {} -> {:?}", wedlm_pred, tokenizer.decode(&[wedlm_pred], false));

    // === Compare logits ===
    let ar_vec: Vec<f32> = ar_last_logits.flatten_all()?.to_vec1()?;
    let wedlm_vec: Vec<f32> = wedlm_mask_logits.flatten_all()?.to_vec1()?;

    let mut max_diff = 0.0f32;
    for (a, w) in ar_vec.iter().zip(wedlm_vec.iter()) {
        let diff = (a - w).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("\nLogits max difference: {:.6}", max_diff);

    // Check if predictions match
    if ar_pred != wedlm_pred {
        // Show top-5 for both
        println!("\nTop-5 AR predictions:");
        let ar_top = ar_probs.flatten_all()?.to_vec1::<f32>()?;
        let mut ar_indexed: Vec<(usize, f32)> = ar_top.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        ar_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (tok, prob) in ar_indexed.iter().take(5) {
            println!("  {} ({:.4}): {:?}", tok, prob, tokenizer.decode(&[*tok as u32], false));
        }

        println!("\nTop-5 WeDLM predictions:");
        let wedlm_top = wedlm_probs.flatten_all()?.to_vec1::<f32>()?;
        let mut wedlm_indexed: Vec<(usize, f32)> = wedlm_top.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        wedlm_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (tok, prob) in wedlm_indexed.iter().take(5) {
            println!("  {} ({:.4}): {:?}", tok, prob, tokenizer.decode(&[*tok as u32], false));
        }
    }

    // The key assertion: predictions should match
    assert_eq!(
        ar_pred, wedlm_pred,
        "INVARIANT B FAILED: Single-mask prediction differs from AR!\n\
         AR predicted: {} -> {:?}\n\
         WeDLM predicted: {} -> {:?}\n\
         This indicates the logits->position mapping is wrong.\n\
         Likely an off-by-one error in position indexing.",
        ar_pred, tokenizer.decode(&[ar_pred], false),
        wedlm_pred, tokenizer.decode(&[wedlm_pred], false)
    );

    println!("\n✓ Invariant B PASSED: Single-mask matches AR prediction\n");

    Ok(())
}

/// Invariant C: Multi-mask first step - position 0 should match AR
///
/// This tests the first step of WeDLM with multiple MASKs.
/// The logits at MASK position 0 should predict the same token as AR.
///
/// If this FAILS: The issue is in how forward_with_positions handles
/// multiple MASKs, or how we extract logits for the first MASK position.
#[test]
#[ignore]
fn test_invariant_c_multi_mask_first_position() -> Result<()> {
    println!("\n=== Invariant C: Multi-mask first position should match AR ===\n");

    let (model, tokenizer, _config, device) = load_test_model()?;

    let prompt = "The quick brown";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenize failed: {}", e)))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let prefix_len = prompt_ids.len();

    println!("Prompt: {:?}", prompt);
    println!("Token IDs: {:?}", prompt_ids);

    // === AR: Get next token prediction ===
    let prefix_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prefix_len), &device)?;
    let (ar_logits, _) = model.forward(&prefix_tensor, 0, None)?;
    let ar_logits = ar_logits.to_dtype(DType::F32)?;
    let ar_last_logits = ar_logits.narrow(1, prefix_len - 1, 1)?.squeeze(1)?;
    let ar_probs = softmax(&ar_last_logits, D::Minus1)?;
    let ar_pred: u32 = ar_probs.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("AR next token: {} -> {:?}", ar_pred, tokenizer.decode(&[ar_pred], false));

    // === WeDLM: Multi-mask block (block_size=4, all MASKs) ===
    let block_size = 4;
    let mut sequence: Vec<i64> = prompt_ids.clone();
    for _ in 0..block_size {
        sequence.push(MASK_TOKEN_ID as i64);
    }
    let seq_len = sequence.len();

    println!("WeDLM input: {:?}", &sequence[prefix_len..]);

    // All positions are MASKs - simulating first step of generate_block
    // With reorder: all MASKs stay at positions prefix_len..prefix_len+block_size
    // Positions: [prefix_len, prefix_len+1, prefix_len+2, prefix_len+3]
    let positions: Vec<i64> = (0..seq_len as i64).collect();
    let positions_tensor = Tensor::from_vec(positions, (1, seq_len), &device)?;
    let input_tensor = Tensor::from_vec(sequence, (1, seq_len), &device)?;

    let (wedlm_logits, _) = model.forward_with_positions(
        &input_tensor,
        &positions_tensor,
        None,
        None,
    )?;
    let wedlm_logits = wedlm_logits.to_dtype(DType::F32)?;

    // Get logits for first MASK position (position prefix_len)
    let first_mask_logits = wedlm_logits.narrow(1, prefix_len, 1)?.squeeze(1)?;
    let first_mask_probs = softmax(&first_mask_logits, D::Minus1)?;
    let first_mask_pred: u32 = first_mask_probs.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("WeDLM first MASK prediction: {} -> {:?}", first_mask_pred, tokenizer.decode(&[first_mask_pred], false));

    // Also check second MASK position - what does it predict?
    let second_mask_logits = wedlm_logits.narrow(1, prefix_len + 1, 1)?.squeeze(1)?;
    let second_mask_probs = softmax(&second_mask_logits, D::Minus1)?;
    let second_mask_pred: u32 = second_mask_probs.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("WeDLM second MASK prediction: {} -> {:?}", second_mask_pred, tokenizer.decode(&[second_mask_pred], false));

    // Show logits comparison
    let ar_vec: Vec<f32> = ar_last_logits.flatten_all()?.to_vec1()?;
    let mask_vec: Vec<f32> = first_mask_logits.flatten_all()?.to_vec1()?;
    let mut max_diff = 0.0f32;
    for (a, m) in ar_vec.iter().zip(mask_vec.iter()) {
        let diff = (a - m).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("\nLogits max diff (AR vs first MASK): {:.6}", max_diff);

    // The first MASK should predict the same as AR
    if ar_pred != first_mask_pred {
        println!("\nTop-5 AR:");
        let ar_top: Vec<f32> = ar_probs.flatten_all()?.to_vec1()?;
        let mut ar_indexed: Vec<(usize, f32)> = ar_top.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        ar_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (tok, prob) in ar_indexed.iter().take(5) {
            println!("  {} ({:.4}): {:?}", tok, prob, tokenizer.decode(&[*tok as u32], false));
        }

        println!("\nTop-5 first MASK:");
        let mask_top: Vec<f32> = first_mask_probs.flatten_all()?.to_vec1()?;
        let mut mask_indexed: Vec<(usize, f32)> = mask_top.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        mask_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (tok, prob) in mask_indexed.iter().take(5) {
            println!("  {} ({:.4}): {:?}", tok, prob, tokenizer.decode(&[*tok as u32], false));
        }
    }

    assert_eq!(
        ar_pred, first_mask_pred,
        "INVARIANT C FAILED: First MASK prediction differs from AR!\n\
         This indicates multi-mask forward doesn't preserve position semantics."
    );

    println!("\n✓ Invariant C PASSED: Multi-mask first position matches AR\n");

    Ok(())
}

/// Invariant D: generate_block-style forward with prefix cache
///
/// This tests EXACTLY what generate_block does:
/// 1. Cache the prefix K/V
/// 2. Pass only the block (not prefix+block) to forward_with_positions
/// 3. Check that predictions match AR
///
/// If this FAILS: The prefix caching path corrupts predictions
#[test]
#[ignore]
fn test_invariant_d_cached_block_forward() -> Result<()> {
    println!("\n=== Invariant D: Cached block forward should match AR ===\n");

    let (model, tokenizer, _config, device) = load_test_model()?;

    let prompt = "The quick brown";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenize failed: {}", e)))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let prefix_len = prompt_ids.len();

    println!("Prompt: {:?}", prompt);
    println!("Token IDs: {:?}", prompt_ids);
    println!("Prefix length: {}", prefix_len);

    // === AR: Get next token prediction ===
    let prefix_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prefix_len), &device)?;
    let (ar_logits, _) = model.forward(&prefix_tensor, 0, None)?;
    let ar_logits = ar_logits.to_dtype(DType::F32)?;
    let ar_last_logits = ar_logits.narrow(1, prefix_len - 1, 1)?.squeeze(1)?;
    let ar_pred: u32 = ar_last_logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("AR next token: {} -> {:?}", ar_pred, tokenizer.decode(&[ar_pred], false));

    // === WeDLM generate_block style ===
    // Step 1: Cache the prefix
    let (_, prefix_caches) = model.forward(&prefix_tensor, 0, None)?;
    let cache_refs: Vec<Option<(Tensor, Tensor)>> = prefix_caches.into_iter().map(Some).collect();

    println!("Prefix cache created ({} layers)", cache_refs.len());

    // Step 2: Create block of all MASKs (like generate_block does)
    let block_size = 4;
    let block_tokens: Vec<i64> = vec![MASK_TOKEN_ID as i64; block_size];

    // Step 3: Positions for block are [prefix_len, prefix_len+1, ...]
    // This is what compute_block_reorder_into produces when all are MASKs
    let positions: Vec<i64> = (0..block_size)
        .map(|i| (prefix_len + i) as i64)
        .collect();

    println!("Block tokens: {:?}", block_tokens);
    println!("Block positions: {:?}", positions);

    let block_tensor = Tensor::from_vec(block_tokens, (1, block_size), &device)?;
    let positions_tensor = Tensor::from_vec(positions, (1, block_size), &device)?;

    // Step 4: Forward ONLY the block with prefix cache
    // This is exactly what generate_block does
    let (block_logits, _) = model.forward_with_positions(
        &block_tensor,
        &positions_tensor,
        Some(&cache_refs),
        None,
    )?;
    let block_logits = block_logits.to_dtype(DType::F32)?;

    println!("Block logits shape: {:?}", block_logits.dims());

    // Get prediction for first position in block (which should predict " fox")
    let first_block_logits = block_logits.narrow(1, 0, 1)?.squeeze(1)?;
    let first_pred: u32 = first_block_logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("First block position prediction: {} -> {:?}", first_pred, tokenizer.decode(&[first_pred], false));

    // Show logits diff
    let ar_vec: Vec<f32> = ar_last_logits.flatten_all()?.to_vec1()?;
    let block_vec: Vec<f32> = first_block_logits.flatten_all()?.to_vec1()?;
    let mut max_diff = 0.0f32;
    for (a, b) in ar_vec.iter().zip(block_vec.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("\nLogits max diff (AR last vs block first): {:.6}", max_diff);

    if ar_pred != first_pred {
        println!("\nPREDICTIONS DIFFER!");
        println!("Top-5 AR:");
        let ar_probs = softmax(&ar_last_logits, D::Minus1)?;
        let ar_top: Vec<f32> = ar_probs.flatten_all()?.to_vec1()?;
        let mut ar_indexed: Vec<(usize, f32)> = ar_top.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        ar_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (tok, prob) in ar_indexed.iter().take(5) {
            println!("  {} ({:.4}): {:?}", tok, prob, tokenizer.decode(&[*tok as u32], false));
        }

        println!("\nTop-5 Block first:");
        let block_probs = softmax(&first_block_logits, D::Minus1)?;
        let block_top: Vec<f32> = block_probs.flatten_all()?.to_vec1()?;
        let mut block_indexed: Vec<(usize, f32)> = block_top.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        block_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (tok, prob) in block_indexed.iter().take(5) {
            println!("  {} ({:.4}): {:?}", tok, prob, tokenizer.decode(&[*tok as u32], false));
        }
    }

    assert_eq!(
        ar_pred, first_pred,
        "INVARIANT D FAILED: Cached block forward differs from AR!\n\
         AR predicted: {} -> {:?}\n\
         Block predicted: {} -> {:?}\n\
         This indicates prefix caching corrupts the forward pass.",
        ar_pred, tokenizer.decode(&[ar_pred], false),
        first_pred, tokenizer.decode(&[first_pred], false)
    );

    println!("\n✓ Invariant D PASSED: Cached block forward matches AR\n");

    Ok(())
}

/// Invariant E: Test actual generate_block output
///
/// This calls the actual generate_block function to see what it produces
#[test]
#[ignore]
fn test_invariant_e_generate_block_output() -> Result<()> {
    println!("\n=== Invariant E: Test actual generate_block output ===\n");

    let (model, tokenizer, _config, device) = load_test_model()?;

    let prompt = "The quick brown";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenize failed: {}", e)))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let prefix_len = prompt_ids.len();

    println!("Prompt: {:?}", prompt);
    println!("Token IDs: {:?}", prompt_ids);

    // === AR: Get expected next tokens ===
    let prefix_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prefix_len), &device)?;

    // Run AR for 4 tokens
    println!("\n--- AR generation (4 tokens) ---");
    let mut ar_tokens: Vec<u32> = Vec::new();
    let mut ar_input = prefix_tensor.clone();
    for i in 0..4 {
        let (ar_logits, _) = model.forward(&ar_input, 0, None)?;
        let ar_logits = ar_logits.to_dtype(DType::F32)?;
        let last_logits = ar_logits.narrow(1, ar_input.dim(1)? - 1, 1)?.squeeze(1)?;
        let pred: u32 = last_logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;
        ar_tokens.push(pred);
        println!("AR token {}: {} -> {:?}", i, pred, tokenizer.decode(&[pred], false));

        // Extend input for next step
        let next_tok = Tensor::from_vec(vec![pred as i64], (1, 1), &device)?;
        ar_input = Tensor::cat(&[&ar_input, &next_tok], 1)?;
    }

    // === WeDLM: Test with different entropy thresholds ===
    for threshold in [0.6, 1.0, 2.0, 4.0] {
        println!("\n--- WeDLM generate_block (entropy_threshold={}) ---", threshold);
        let mut decoder = WeDLMDecoder::new(&model, None);
        let params = SamplingParams {
            temperature: 0.0,
            entropy_threshold: threshold,
            ..Default::default()
        };

        let (_, block_tokens_test, stats_test) = decoder.generate_block(&prefix_tensor, 4, &params)?;
        let block_u32: Vec<u32> = block_tokens_test.iter().map(|&x| x as u32).collect();

        let matches: usize = block_u32.iter().zip(&ar_tokens).filter(|(a, b)| a == b).count();
        println!("Output: {:?}", block_tokens_test.iter().map(|&t| tokenizer.decode(&[t as u32], false).unwrap_or_default()).collect::<Vec<_>>());
        println!("Steps: {}, Parallelism: {:.1}x, Matches: {}/4, Avg entropy: {:.4}",
            stats_test.steps,
            stats_test.tokens_generated as f32 / stats_test.steps as f32,
            matches,
            stats_test.avg_entropy);
    }

    // Use threshold=2.0 for final comparison
    let mut decoder = WeDLMDecoder::new(&model, None);
    let params = SamplingParams {
        temperature: 0.0,
        entropy_threshold: 2.0,
        ..Default::default()
    };

    let (output_ids, block_tokens, stats) = decoder.generate_block(&prefix_tensor, 4, &params)?;

    println!("\nBlock tokens: {:?}", block_tokens);
    let block_tokens_u32: Vec<u32> = block_tokens.iter().map(|&x| x as u32).collect();
    for (i, &tok) in block_tokens_u32.iter().enumerate() {
        println!("Block token {}: {} -> {:?}", i, tok, tokenizer.decode(&[tok], false));
    }

    println!("\nStats: steps={}, tokens_generated={}, avg_entropy={:.4}, max_entropy={:.4}",
        stats.steps, stats.tokens_generated, stats.avg_entropy, stats.max_entropy);
    println!("Parallelism: {:.1}x (tokens/steps)", stats.tokens_generated as f32 / stats.steps as f32);

    // === Compare ===
    println!("\n--- Comparison ---");
    for i in 0..4 {
        let ar_tok = ar_tokens[i];
        let wedlm_tok = block_tokens_u32[i];
        let match_str = if ar_tok == wedlm_tok { "✓" } else { "✗ MISMATCH" };
        println!("Position {}: AR={} WeDLM={} {}", i, ar_tok, wedlm_tok, match_str);
    }

    // Check if at least first token matches
    assert_eq!(
        ar_tokens[0], block_tokens_u32[0],
        "INVARIANT E FAILED: First block token differs from AR!\n\
         AR: {} -> {:?}\n\
         WeDLM: {} -> {:?}",
        ar_tokens[0], tokenizer.decode(&[ar_tokens[0]], false),
        block_tokens_u32[0], tokenizer.decode(&[block_tokens_u32[0]], false)
    );

    println!("\n✓ Invariant E: First token matches AR\n");

    Ok(())
}

/// Additional diagnostic: Test that prefix caching doesn't change predictions
#[test]
#[ignore]
fn test_prefix_cache_consistency() -> Result<()> {
    println!("\n=== Diagnostic: Prefix cache consistency ===\n");

    let (model, tokenizer, _config, device) = load_test_model()?;

    let prompt = "The quick brown fox jumps";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenize failed: {}", e)))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let prompt_len = prompt_ids.len();

    println!("Prompt: {:?} ({} tokens)", prompt, prompt_len);

    // Split into prefix and continuation
    let split_at = prompt_len - 2;
    let prefix_ids: Vec<i64> = prompt_ids[..split_at].to_vec();
    let continuation_ids: Vec<i64> = prompt_ids[split_at..].to_vec();

    println!("Prefix: {} tokens, Continuation: {} tokens", prefix_ids.len(), continuation_ids.len());

    // === Full forward pass (no cache) ===
    let full_input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), &device)?;
    let (full_logits, _) = model.forward(&full_input, 0, None)?;
    let full_logits = full_logits.to_dtype(DType::F32)?;
    let full_last = full_logits.narrow(1, prompt_len - 1, 1)?.squeeze(1)?;
    let full_pred: u32 = full_last.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("Full forward prediction: {} -> {:?}", full_pred, tokenizer.decode(&[full_pred], false));

    // === With prefix caching ===
    // Step 1: Cache the prefix
    let prefix_input = Tensor::from_vec(prefix_ids.clone(), (1, split_at), &device)?;
    let (_, prefix_caches) = model.forward(&prefix_input, 0, None)?;

    // Step 2: Continue with cached prefix
    let cont_input = Tensor::from_vec(continuation_ids, (1, prompt_len - split_at), &device)?;
    let positions: Vec<i64> = (split_at as i64..prompt_len as i64).collect();
    let positions_tensor = Tensor::from_vec(positions, (1, prompt_len - split_at), &device)?;

    let cache_refs: Vec<Option<(Tensor, Tensor)>> = prefix_caches.into_iter().map(Some).collect();
    let (cached_logits, _) = model.forward_with_positions(
        &cont_input,
        &positions_tensor,
        Some(&cache_refs),
        None,
    )?;
    let cached_logits = cached_logits.to_dtype(DType::F32)?;
    let cached_last = cached_logits.narrow(1, (prompt_len - split_at) - 1, 1)?.squeeze(1)?;
    let cached_pred: u32 = cached_last.argmax(D::Minus1)?.squeeze(0)?.to_scalar()?;

    println!("Cached forward prediction: {} -> {:?}", cached_pred, tokenizer.decode(&[cached_pred], false));

    assert_eq!(
        full_pred, cached_pred,
        "CACHE CONSISTENCY FAILED: Predictions differ with/without cache!\n\
         This indicates prefix caching is corrupting the KV state."
    );

    println!("\n✓ Prefix cache consistency PASSED\n");

    Ok(())
}
