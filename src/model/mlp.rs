//! WeDLM MLP with SiLU (Swish) gating
//!
//! Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::WeDLMConfig;

/// SiLU-gated MLP (also known as SwiGLU variant)
pub struct WeDLMMLP {
    /// Gate projection: [hidden_size -> intermediate_size]
    gate_proj: Linear,
    /// Up projection: [hidden_size -> intermediate_size]
    up_proj: Linear,
    /// Down projection: [intermediate_size -> hidden_size]
    down_proj: Linear,
}

impl WeDLMMLP {
    pub fn new(config: &WeDLMConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for WeDLMMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() -> Result<()> {
        use candle_core::Device;

        // SiLU(x) = x * sigmoid(x)
        let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], (3,), &Device::Cpu)?;
        let result = candle_nn::ops::silu(&x)?;
        let result_vec: Vec<f32> = result.to_vec1()?;

        // silu(0) = 0
        assert!((result_vec[0] - 0.0).abs() < 1e-6);
        // silu(1) ≈ 0.731
        assert!((result_vec[1] - 0.731).abs() < 0.01);
        // silu(-1) ≈ -0.269
        assert!((result_vec[2] - (-0.269)).abs() < 0.01);

        Ok(())
    }
}
