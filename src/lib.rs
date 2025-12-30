//! WeDLM-rs: Rust inference engine for WeDLM-8B using Candle
//!
//! This crate provides a high-performance implementation of the WeDLM
//! (Weighted Diffusion Language Model) architecture with block decoding
//! for parallel token generation.

pub mod cache;
pub mod config;
pub mod decoding;
pub mod engine;
pub mod model;
pub mod weights;

pub use config::WeDLMConfig;
pub use engine::WeDLMEngine;

/// MASK token ID for WeDLM decoding
/// From Tencent's WeDLM implementation: wedlm/engine/model_runner.py:_init_mask_token
/// This is vocab_size (151643) + 22 added tokens = 151665 (last token + 1)
pub const MASK_TOKEN_ID: u32 = 151665;

/// Default block size for WeDLM generation
/// Sweep testing shows block_size=32 with aggressive acceptance gives ~20x speedup
pub const DEFAULT_BLOCK_SIZE: usize = 32;
