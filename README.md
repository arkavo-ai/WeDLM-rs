# WeDLM-rs

An independent, high-performance Rust implementation of the WeDLM architecture.

This project provides a native Rust inference engine for WeDLM models, optimized for Apple Silicon with Metal acceleration.

## Attribution

This implementation is based on the WeDLM architecture developed by Tencent:

- **Original Repository**: [Tencent/WeDLM](https://github.com/Tencent/WeDLM)
- **Paper**: [WeDLM: Weighted Diffusion Language Model](https://arxiv.org/abs/2505.18567)

We thank the Tencent research team for their pioneering work on parallel decoding with diffusion language models.

## Features

- Pure Rust implementation using the [Candle](https://github.com/huggingface/candle) ML framework
- Metal acceleration for Apple Silicon (M1/M2/M3/M4)
- F16 precision for optimal memory/performance balance
- Compatible with HuggingFace WeDLM model weights

## Model Architecture

WeDLM-8B specifications:
- 8.19B parameters
- 36 transformer layers
- 4096 hidden dimension
- 32 attention heads (8 KV heads with GQA)
- QK-normalization enabled
- RoPE with theta=1,000,000

## Quick Start

### Prerequisites

- Rust 1.75+
- macOS with Apple Silicon (for Metal support)
- WeDLM model weights from HuggingFace

### Installation

```bash
git clone https://github.com/arkavo-ai/WeDLM-rs.git
cd WeDLM-rs
cargo build --features metal --release
```

### Download Model

```bash
# Using huggingface-cli
huggingface-cli download tencent/WeDLM-8B-Instruct
```

### Usage

```bash
# Test model loading and forward pass
cargo run --features metal --release --bin wedlm-cli -- test \
  --model ~/.cache/huggingface/hub/models--tencent--WeDLM-8B-Instruct/snapshots/<snapshot-id>

# Generate text (autoregressive mode)
cargo run --features metal --release --bin wedlm-cli -- generate \
  --model ~/.cache/huggingface/hub/models--tencent--WeDLM-8B-Instruct/snapshots/<snapshot-id> \
  --prompt "Hello, world!" \
  --max-tokens 64 \
  --autoregressive
```

## Project Structure

```
src/
├── lib.rs              # Library root
├── config.rs           # Model configuration
├── engine.rs           # High-level inference engine
├── weights.rs          # Safetensor weight loading
├── cache.rs            # KV cache implementation
├── model/
│   ├── mod.rs          # Model module
│   ├── attention.rs    # Multi-head attention with GQA
│   ├── mlp.rs          # SwiGLU MLP
│   ├── layer.rs        # Transformer decoder layer
│   ├── backbone.rs     # Transformer backbone
│   ├── causal_lm.rs    # Causal LM head
│   └── rope.rs         # Rotary position embeddings
└── decoding/
    ├── mod.rs          # Decoding module
    ├── sampler.rs      # Token sampling
    ├── wedlm.rs        # WeDLM block decoding
    └── reorder.rs      # Topological reordering
```

## Development Status

- [x] Model architecture implementation
- [x] Weight loading from safetensors
- [x] Metal acceleration with F16
- [x] Forward pass verification
- [ ] Full WeDLM parallel decoding
- [ ] KV cache optimization
- [ ] Benchmarking suite

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

This project is licensed under the same terms as the original WeDLM implementation.
