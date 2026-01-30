# WeDLM-rs

A Rust implementation of Tencent's [WeDLM](https://github.com/Tencent/WeDLM) parallel decoding architecture.

## Quick Start (5 minutes)

### 1. Install Rust (if needed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Install HuggingFace CLI (if needed)

```bash
brew install huggingface-cli
```

### 3. Download the Model (~16GB)

```bash
hf download tencent/WeDLM-8B-Instruct
```

### 4. Clone and Build

```bash
git clone https://github.com/arkavo-ai/WeDLM-rs.git
cd WeDLM-rs
cargo build --release
```

### 5. Run!

```bash
target/release/wedlm-cli generate \
  --model tencent/WeDLM-8B-Instruct \
  --prompt "Explain quantum computing in simple terms"
```

**That's it!** You should see output like:

```
Resolving HuggingFace model: tencent/WeDLM-8B-Instruct
Model loading...
Model loaded in 3.42s

Generating...

--- Output ---
Explain quantum computing in simple terms: Traditional computers use bits...

--- Stats ---
Time: 11.3s
Tokens generated: 128
Speed: 11.3 tok/s
```

---

## Usage Examples

### Basic Generation

```bash
# Short response
cargo run --release -- generate \
  -m tencent/WeDLM-8B-Instruct \
  -p "What is the capital of France?" \
  -n 32

# Longer response
cargo run --release -- generate \
  -m tencent/WeDLM-8B-Instruct \
  -p "Write a haiku about programming:" \
  -n 64
```

### Compare Speed: WeDLM vs Standard

```bash
# Run benchmark
cargo run --release -- benchmark \
  -m tencent/WeDLM-8B-Instruct \
  -n 128

# You'll see something like:
# Autoregressive:  14.1s avg (4.5 tok/s)
# WeDLM Parallel:   5.7s avg (11.3 tok/s)
# WeDLM is 2.5x FASTER than autoregressive
```

### All Options

```bash
cargo run --release -- generate --help
```

| Option             | Short | Default  | Description                |
|--------------------|-------|----------|----------------------------|
| `--model`          | `-m`  | required | Model ID or local path     |
| `--prompt`         | `-p`  | required | Your input text            |
| `--max-tokens`     | `-n`  | 128      | Max tokens to generate     |
| `--temperature`    | `-t`  | 0.2      | Creativity (0.0-1.0)       |
| `--block-size`     | `-b`  | 96       | Parallel block size        |
| `--autoregressive` |       | false    | Use slow standard decoding |

---

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4) — Metal GPU acceleration
- **16GB+ RAM** — Model uses ~16GB
- **Rust 1.75+** — Install via rustup

> **Note**: CPU-only and CUDA support are possible but not the default. See [Building](#building) below.

---

## How It Works

WeDLM generates multiple tokens in parallel instead of one at a time:

```
Standard (autoregressive):  Token → Token → Token → Token  (slow)
WeDLM (parallel):           [Token Token Token Token]      (fast!)
```

The model predicts a block of tokens simultaneously, then refines them. This achieves **~2.5x speedup** over standard autoregressive decoding on Apple Silicon.

---

## Building

### Default (Apple Silicon with Metal)

```bash
cargo build --release
```

### CPU Only

```bash
cargo build --release --no-default-features
```

### NVIDIA GPU (CUDA)

```bash
cargo build --release --no-default-features --features cuda
```

---

## Performance Tuning

### Benchmarks

| Hardware | GPU Cores | RAM | Preset | WeDLM | AR | Speedup |
|----------|-----------|-----|--------|-------|-----|---------|
| M1 Max | 32 | 32GB | Balanced | 11.3 tok/s | 4.5 tok/s | 2.5x |
| M4 Max | 40 | 128GB | Balanced | 22.4 tok/s | 5.8 tok/s | 3.9x |
| M4 Max | 40 | 128GB | Fast | 71.2 tok/s | 5.8 tok/s | 12.2x |

### Presets

Presets are tuned for Apple Silicon unified memory architecture:

| Preset | Entropy (τ) | Lambda (λ) | Max/Step | Speed | Quality |
|--------|-------------|------------|----------|-------|---------|
| `quality` | 0.5 | 0.10 | 4 | ~18 tok/s | Best |
| `balanced` | 0.8 | 0.05 | 8 | ~22 tok/s | Good |
| `fast` | 1.0 | 0.05 | 6 | ~71 tok/s | Acceptable |

```bash
# Use fast preset for maximum speed
./target/release/wedlm-cli generate \
  -m tencent/WeDLM-8B-Instruct \
  -p "Your prompt" \
  --preset fast
```

### Theoretical Limits

On Apple Silicon with unified memory:
- Model size: 8B params × 2 bytes (FP16) = 16 GB
- M4 Max bandwidth: ~400 GB/s
- **Theoretical max: ~150-200 tok/s** (limited by memory bandwidth)

To exceed this, quantization (INT4/INT8) would be required.

---

## Attribution

Based on the WeDLM architecture by Tencent:

- **Paper**: [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737)
- **Original**: [github.com/Tencent/WeDLM](https://github.com/Tencent/WeDLM)

---

## License

Apache License 2.0
