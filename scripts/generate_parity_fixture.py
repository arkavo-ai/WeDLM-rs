#!/usr/bin/env python3
"""Generate a tiny WeDLM parity fixture without torch/triton.

This script uses NumPy only and writes a JSON fixture with:
- config
- weights
- input_ids + explicit positions
- expected logits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    x = x / np.sqrt(var + eps)
    return (x * weight).astype(np.float32)


def linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # weight: [out, in] -> x @ weight.T
    return np.matmul(x, weight.T)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(q: np.ndarray, k: np.ndarray, positions: np.ndarray, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    pos = positions.reshape(-1).astype(np.float32)
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.outer(pos, inv_freq).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)[None, None, :, :]
    sin = np.sin(emb).astype(np.float32)[None, None, :, :]

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def causal_mask(seq_len: int) -> np.ndarray:
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i, j] = -np.inf
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tests/fixtures/wedlm_parity_small.json",
        help="Output fixture path",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(0)

    config = {
        "vocab_size": 16,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 12,
        "hidden_act": "silu",
        "max_position_embeddings": 16,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "qk_norm": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "mask_token_id": None,
        "dtype": "float32",
        "rope_scaling": None,
    }

    batch = 1
    seq_len = 3
    input_ids = np.array([[1, 2, 3]], dtype=np.int64)
    positions = np.array([[0, 2, 5]], dtype=np.int64)

    # Weights (deterministic, small values)
    def rand(shape):
        return rng.uniform(-0.1, 0.1, size=shape).astype(np.float32)

    weights = {}
    weights["model.embed_tokens.weight"] = rand((config["vocab_size"], config["hidden_size"]))

    weights["model.layers.0.self_attn.q_proj.weight"] = rand((config["num_attention_heads"] * config["head_dim"], config["hidden_size"]))
    weights["model.layers.0.self_attn.k_proj.weight"] = rand((config["num_key_value_heads"] * config["head_dim"], config["hidden_size"]))
    weights["model.layers.0.self_attn.v_proj.weight"] = rand((config["num_key_value_heads"] * config["head_dim"], config["hidden_size"]))
    weights["model.layers.0.self_attn.o_proj.weight"] = rand((config["num_attention_heads"] * config["head_dim"], config["hidden_size"]))

    weights["model.layers.0.self_attn.q_norm.weight"] = np.ones((config["head_dim"],), dtype=np.float32)
    weights["model.layers.0.self_attn.k_norm.weight"] = np.ones((config["head_dim"],), dtype=np.float32)

    weights["model.layers.0.mlp.gate_proj.weight"] = rand((config["intermediate_size"], config["hidden_size"]))
    weights["model.layers.0.mlp.up_proj.weight"] = rand((config["intermediate_size"], config["hidden_size"]))
    weights["model.layers.0.mlp.down_proj.weight"] = rand((config["hidden_size"], config["intermediate_size"]))

    weights["model.layers.0.input_layernorm.weight"] = np.ones((config["hidden_size"],), dtype=np.float32)
    weights["model.layers.0.post_attention_layernorm.weight"] = np.ones((config["hidden_size"],), dtype=np.float32)
    weights["model.norm.weight"] = np.ones((config["hidden_size"],), dtype=np.float32)

    weights["lm_head.weight"] = rand((config["vocab_size"], config["hidden_size"]))

    # Forward pass (Rust architecture)
    hidden = weights["model.embed_tokens.weight"][input_ids]

    # Layer 0
    residual = hidden
    hidden = rms_norm(hidden, weights["model.layers.0.input_layernorm.weight"], config["rms_norm_eps"])

    q = linear(hidden, weights["model.layers.0.self_attn.q_proj.weight"])
    k = linear(hidden, weights["model.layers.0.self_attn.k_proj.weight"])
    v = linear(hidden, weights["model.layers.0.self_attn.v_proj.weight"])

    q = q.reshape(batch, seq_len, config["num_attention_heads"], config["head_dim"])
    k = k.reshape(batch, seq_len, config["num_key_value_heads"], config["head_dim"])
    v = v.reshape(batch, seq_len, config["num_key_value_heads"], config["head_dim"])

    if config["qk_norm"]:
        q = rms_norm(q, weights["model.layers.0.self_attn.q_norm.weight"], config["rms_norm_eps"])
        k = rms_norm(k, weights["model.layers.0.self_attn.k_norm.weight"], config["rms_norm_eps"])

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    q, k = apply_rope(q, k, positions, config["head_dim"], config["rope_theta"])

    if config["num_attention_heads"] != config["num_key_value_heads"]:
        groups = config["num_attention_heads"] // config["num_key_value_heads"]
        k = np.repeat(k, groups, axis=1)
        v = np.repeat(v, groups, axis=1)

    scale = np.float32(1.0 / np.sqrt(config["head_dim"]))
    attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    attn = attn + causal_mask(seq_len)[None, None, :, :]
    probs = softmax(attn, axis=-1)
    attn_out = np.matmul(probs, v)

    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, config["hidden_size"])
    attn_out = linear(attn_out, weights["model.layers.0.self_attn.o_proj.weight"])

    hidden = residual + attn_out

    residual = hidden
    hidden = rms_norm(hidden, weights["model.layers.0.post_attention_layernorm.weight"], config["rms_norm_eps"])

    gate = silu(linear(hidden, weights["model.layers.0.mlp.gate_proj.weight"]))
    up = linear(hidden, weights["model.layers.0.mlp.up_proj.weight"])
    mlp_out = linear(gate * up, weights["model.layers.0.mlp.down_proj.weight"])

    hidden = residual + mlp_out

    hidden = rms_norm(hidden, weights["model.norm.weight"], config["rms_norm_eps"])
    logits = linear(hidden, weights["lm_head.weight"])

    fixture = {
        "config": config,
        "input_ids": input_ids.tolist(),
        "positions": positions.tolist(),
        "weights": {
            name: {"shape": list(w.shape), "data": w.flatten().tolist()}
            for name, w in weights.items()
        },
        "expected_logits": {"shape": list(logits.shape), "data": logits.flatten().tolist()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote fixture to {output_path}")


if __name__ == "__main__":
    main()
