# WeDLM-rs Baseline Profile

**Date**: 2025-12-31
**Commit**: 895a4ea (Batch probs readback)
**Hardware**: Mac Studio M1 Ultra

## Benchmark Results

| Mode | Throughput | Speedup |
|------|------------|---------|
| Autoregressive | 2.6 tok/s | 1x |
| WeDLM Parallel | 12.5 tok/s | 4.8x |

## Profile: Time Profiler (AR inference phase)

Captured via `sample` during autoregressive benchmark.

### CPU Time Distribution

| Location | Samples | % | Notes |
|----------|---------|---|-------|
| `waitUntilCompleted` | 7591 | 98.1% | Blocked on GPU |
| Forward pass | 144 | 1.9% | RoPE, attention, MLP |
| Buffer allocation | ~30 | <1% | Metal buffer creation |

### Call Stack (top of stack summary)

```
__psynch_cvwait (GPU wait)           7591
RotaryEmbedding::apply                 28
WeDLMDecoderLayer::forward             12
allocate_buffer (Metal)                30
mach_msg2_trap (IOKit)                 24
```

### Key Findings

1. **GPU-bound, not CPU-bound**: 98% of CPU time is waiting for GPU completion
2. **No CPU submission stalls**: Metal command encoding is not the bottleneck
3. **Buffer allocation minimal**: Per-tensor buffer creation is negligible
4. **`to_cpu_storage` triggers sync**: Every `to_vec1()`/`to_vec2()` call blocks

### Implications for Optimization

- Reducing GPU sync count (our optimization) helps, but main cost is GPU compute
- Further gains require reducing forward passes (algorithmic) or GPU kernel optimization
- Batch probs readback (K syncs → 1) is correct approach but bounded by GPU time

## Profile: Metal System Trace

Command buffer operations from wedlm-cli during inference:

| Operation | Typical Duration | Encoder Time |
|-----------|------------------|--------------|
| `to_dtype` | 350-450 µs | 35-65 µs |
| `binary` | 350-430 µs | 25-72 µs |
| `matmul` | 360-460 µs | 57-62 µs |
| `copy_strided` | 360-400 µs | 57-67 µs |
| `uneg` | 370-410 µs | 60-85 µs |
| `usilu` | 360-410 µs | 60-63 µs |
| `usqr` | 360-460 µs | 55-67 µs |
| `usqrt` | 360-410 µs | 55-62 µs |
| `to_cpu` | 1.5-2.0 s | 350 µs |

### GPU Observations

1. **Many small command buffers**: Each tensor op dispatches its own command buffer (~350-450 µs each)
2. **`to_cpu` dominates**: CPU readback takes 1.5-2s (the `waitUntilCompleted` blocking)
3. **Encoder time is minimal**: 35-85 µs per op (CPU encoding not a bottleneck)
4. **No batching of ops**: Candle dispatches each op separately (potential optimization)

### GPU Utilization Pattern

- During model loading: Large command buffers (2-6s) for weight transfers
- During AR inference: Continuous small command buffers, CPU blocked on `to_cpu`
- During WeDLM: Similar pattern but fewer total syncs due to parallel decoding

## Trace Files

- `profiles/baseline-time-profiler.trace` - Instruments Time Profiler trace
- `profiles/baseline-metal-trace.trace` - Metal System Trace (GPU)
- `/tmp/wedlm-cli_*.sample.txt` - Raw sample output
