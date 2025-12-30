//! WeDLM Block Decoding
//!
//! Implements the parallel decoding algorithm with topological reordering.
//! 
//! Production safeguards:
//! - Adaptive block sizing based on entropy
//! - Automatic cache refresh for long sequences  
//! - Hard entropy limits that force conservative decoding

pub mod reorder;
pub mod sampler;
pub mod wedlm;

pub use reorder::topological_reorder;
pub use sampler::SamplingParams;
pub use wedlm::{
    BlockStats, WeDLMDecoder,
    ENTROPY_SOFT_THRESHOLD, ENTROPY_HARD_THRESHOLD,
    MAX_CACHED_LENGTH, CACHE_CHECK_INTERVAL,
};
