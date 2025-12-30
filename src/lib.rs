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

/// MASK token ID for WeDLM decoding (from WeDLM tokenizer)
pub const MASK_TOKEN_ID: u32 = 151666;

/// Default block size for WeDLM generation
/// Sweep testing shows block_size=32 with aggressive acceptance gives ~20x speedup
pub const DEFAULT_BLOCK_SIZE: usize = 32;
