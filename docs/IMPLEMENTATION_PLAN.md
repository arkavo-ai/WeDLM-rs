# WeDLM-rs Implementation Plan

## Overview

High-performance Rust inference engine for WeDLM-8B using Candle, targeting Apple Silicon with Metal acceleration.

## Completed

### Phase 1: Core Architecture
- [x] Project structure and Cargo.toml
- [x] Model configuration (WeDLMConfig)
- [x] Weight loading from sharded safetensors (4 shards, 399 tensors)
- [x] Token embeddings with CPU fallback for Metal compatibility
- [x] RMSNorm (computed in F32 for numerical stability)
- [x] Rotary Position Embeddings (RoPE, theta=1,000,000)
- [x] Multi-head attention with Grouped Query Attention (32 heads → 8 KV heads)
- [x] QK-normalization (critical: applied BEFORE transpose, BEFORE RoPE)
- [x] SwiGLU MLP (gate_proj, up_proj, down_proj)
- [x] Transformer decoder layers (36 layers)
- [x] Causal LM head
- [x] Forward pass verification on Metal with F16

### Phase 2: Basic Inference
- [x] Tokenizer integration (HuggingFace tokenizers)
- [x] High-level WeDLMEngine API
- [x] Autoregressive generation (token-by-token)
- [x] CLI with generate and test commands

### Metal Compatibility Fixes
- [x] CPU embedding lookup (Metal lacks F16 index_select)
- [x] U32 indices for all index operations
- [x] F16 precision throughout (BF16 has gaps in Metal support)

## Remaining Work

### Phase 3: WeDLM Block Decoding
- [ ] **Topological reordering integration**
  - Reorder input sequence: known tokens first, MASK tokens last
  - Build permutation and inverse permutation mappings
  - Apply to KV cache during generation

- [ ] **Block-parallel generation**
  - Generate multiple tokens per forward pass
  - Append MASK tokens to prompt
  - Predict all MASK positions simultaneously

- [ ] **Confidence-based acceptance**
  - Compute per-token confidence from softmax probabilities
  - Accept tokens above confidence threshold
  - Re-mask low-confidence positions for next iteration

- [ ] **Iterative refinement loop**
  - Continue until all positions accepted or max iterations
  - Handle partial acceptance (some tokens accepted, others re-masked)

### Phase 4: KV Cache Optimization
- [ ] **Incremental KV cache**
  - Cache keys/values from previous forward passes
  - Only compute attention for new tokens
  - Proper cache management for reordering

- [ ] **Cache reordering for WeDLM**
  - Reorder cached KV states when permutation changes
  - Handle cache invalidation on re-masking

- [ ] **Memory optimization**
  - Pre-allocate cache for max sequence length
  - Efficient cache update patterns

### Phase 5: Performance & Polish
- [ ] **Benchmarking suite**
  - Tokens per second measurement
  - Memory usage tracking
  - Comparison: autoregressive vs WeDLM block decoding

- [ ] **Batch processing**
  - Support batch_size > 1
  - Efficient batched attention

- [ ] **Streaming output**
  - Yield tokens as they become available
  - Real-time generation feedback

- [ ] **Error handling improvements**
  - Graceful degradation on OOM
  - Better error messages

### Phase 6: Advanced Features (Future)
- [ ] **Speculative decoding integration**
  - Combine WeDLM with draft models
  - Hybrid decoding strategies

- [ ] **Quantization support**
  - INT8/INT4 weight quantization
  - Reduced memory footprint

- [ ] **Multi-GPU support**
  - Tensor parallelism for larger models
  - Pipeline parallelism

## Technical Notes

### Critical Implementation Details

1. **QK-Norm Order** (MUST match exactly):
   ```
   Project → Reshape → QK-NORM → Transpose → RoPE → Attention
   ```
   Wrong order causes high entropy and incoherent generation.

2. **RoPE Parameters**:
   - theta = 1,000,000 (high for long context)
   - head_dim = 128 → 64 inverse frequencies
   - Formula: `inv_freq[i] = 1.0 / (theta ^ (2*i / head_dim))`

3. **Metal Limitations**:
   - No F16/BF16 index_select → use CPU for embeddings
   - Use U32 for all index tensors
   - F16 preferred over BF16 (better Metal support)

4. **MASK Token**: ID 151666 (already in WeDLM tokenizer)

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | 8.19B |
| Layers | 36 |
| Hidden Size | 4096 |
| Attention Heads | 32 |
| KV Heads | 8 (GQA) |
| Head Dim | 128 |
| Intermediate Size | 12288 |
| Vocab Size | 151,936 |
| Max Position | 16,384 |
| RoPE Theta | 1,000,000 |

## Resources

- [Tencent/WeDLM Repository](https://github.com/Tencent/WeDLM)
- [WeDLM Paper](https://arxiv.org/abs/2505.18567)
- [Candle Framework](https://github.com/huggingface/candle)
- [WeDLM-8B-Instruct on HuggingFace](https://huggingface.co/tencent/WeDLM-8B-Instruct)
