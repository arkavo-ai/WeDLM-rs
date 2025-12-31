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
Time: 10.5s
Tokens generated: 128
Speed: 12.2 tok/s
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
# Autoregressive:  50.0s avg (2.6 tok/s)
# WeDLM Parallel:  10.5s avg (12.2 tok/s)
# WeDLM is 4.8x FASTER than autoregressive
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

The model predicts a block of tokens simultaneously, then refines them. This achieves **~4.8x speedup** over standard autoregressive decoding on Apple Silicon.

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

## Attribution

Based on the WeDLM architecture by Tencent:

- **Paper**: [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737)
- **Original**: [github.com/Tencent/WeDLM](https://github.com/Tencent/WeDLM)

---

## License

Apache License 2.0
