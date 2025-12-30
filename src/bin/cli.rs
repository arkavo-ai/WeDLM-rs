//! WeDLM-rs CLI
//!
//! Command-line interface for WeDLM inference.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use wedlm_rs::decoding::SamplingParams;
use wedlm_rs::WeDLMEngine;

#[derive(Parser)]
#[command(name = "wedlm-cli")]
#[command(about = "WeDLM-8B inference engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt
    Generate {
        /// Path to model directory
        #[arg(short, long)]
        model: PathBuf,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum new tokens to generate
        #[arg(short = 'n', long, default_value = "128")]
        max_tokens: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.2")]
        temperature: f32,

        /// Confidence threshold for WeDLM decoding
        #[arg(short, long, default_value = "0.8")]
        confidence: f32,

        /// Use simple autoregressive decoding instead of WeDLM
        #[arg(long)]
        autoregressive: bool,
    },

    /// Test model loading
    Test {
        /// Path to model directory
        #[arg(short, long)]
        model: PathBuf,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            confidence,
            autoregressive,
        } => {
            tracing::info!("Loading model from {:?}...", model);
            let engine = WeDLMEngine::from_pretrained(&model)?;

            tracing::info!("Generating...");
            let output = if autoregressive {
                engine.generate_autoregressive(&prompt, max_tokens, temperature)?
            } else {
                let params = SamplingParams {
                    temperature,
                    confidence_threshold: confidence,
                    ..Default::default()
                };
                engine.generate(&prompt, max_tokens, Some(params))?
            };

            println!("\n{}", output);
        }

        Commands::Test { model } => {
            tracing::info!("Testing model loading from {:?}...", model);

            // Load config
            let config_path = model.join("config.json");
            let config = wedlm_rs::WeDLMConfig::from_file(&config_path)?;
            tracing::info!("Config loaded: {} layers, {} hidden", config.num_hidden_layers, config.hidden_size);

            // Detect device - use F16 on Metal (full support including index_select)
            let (device, dtype) = {
                #[cfg(feature = "metal")]
                {
                    if candle_core::utils::metal_is_available() {
                        tracing::info!("Using Metal device with F16");
                        (candle_core::Device::new_metal(0)?, candle_core::DType::F16)
                    } else {
                        tracing::info!("Metal not available, using CPU with F32");
                        (candle_core::Device::Cpu, candle_core::DType::F32)
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    tracing::info!("Using CPU with F32");
                    (candle_core::Device::Cpu, candle_core::DType::F32)
                }
            };

            tracing::info!("Loading weights...");
            let vb = wedlm_rs::weights::load_model_vb(&model, dtype, &device)?;
            tracing::info!("Weights loaded successfully");

            // Build model
            tracing::info!("Building model...");
            let _model = wedlm_rs::model::WeDLMForCausalLM::new(&config, vb, &device)?;
            tracing::info!("Model built successfully!");

            // Test forward pass with dummy input
            tracing::info!("Testing forward pass...");
            let input = candle_core::Tensor::from_vec(vec![1i64, 2, 3], (1, 3), &device)?;
            let (logits, _) = _model.forward(&input, 0, None)?;
            tracing::info!("Forward pass output shape: {:?}", logits.dims());

            // Show some stats about the output
            let vocab_size = logits.dim(2)?;
            tracing::info!("Vocab size from logits: {}", vocab_size);

            println!("All tests passed!");
        }
    }

    Ok(())
}
