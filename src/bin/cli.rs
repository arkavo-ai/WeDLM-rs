//! WeDLM-rs CLI
//!
//! Command-line interface for WeDLM inference.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use wedlm_rs::decoding::SamplingParams;
use wedlm_rs::WeDLMEngine;

/// Resolve a model path - either local directory or HuggingFace model ID
/// 
/// If the path exists locally, use it directly.
/// Otherwise, treat it as a HF model ID (e.g., "tencent/WeDLM-8B-Instruct")
/// and resolve from the HuggingFace cache.
fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let local_path = PathBuf::from(model);
    
    // Check if it's a local path that exists
    if local_path.exists() {
        return Ok(local_path);
    }
    
    // Treat as HuggingFace model ID
    if !model.contains('/') {
        anyhow::bail!(
            "Model '{}' not found locally and doesn't look like a HF model ID (expected 'org/model')",
            model
        );
    }
    
    eprintln!("Resolving HuggingFace model: {}", model);
    
    let api = Api::new().context("Failed to initialize HuggingFace API")?;
    let repo = api.repo(Repo::new(model.to_string(), RepoType::Model));
    
    // Get the cache directory for this model by fetching config.json
    // This ensures the model is cached and gives us the snapshot path
    let config_path = repo
        .get("config.json")
        .context("Failed to fetch config.json - model may not exist or not be cached")?;
    
    // The model directory is the parent of config.json
    let model_dir = config_path
        .parent()
        .context("Invalid cache path structure")?
        .to_path_buf();
    
    eprintln!("Resolved to: {}", model_dir.display());
    Ok(model_dir)
}

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
        /// Model path or HuggingFace ID (e.g., tencent/WeDLM-8B-Instruct)
        #[arg(short, long)]
        model: String,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum new tokens to generate
        #[arg(short = 'n', long, default_value = "128")]
        max_tokens: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.2")]
        temperature: f32,

        /// Entropy threshold for WeDLM decoding (lower = better quality, higher = faster)
        #[arg(short = 'e', long, default_value = "0.6")]
        entropy_threshold: f32,

        /// Margin threshold (unused with streaming, kept for compatibility)
        #[arg(long, default_value = "0.0")]
        margin_threshold: f32,

        /// Distance penalty coefficient λ for H̃ = H + λ·d selection
        #[arg(long, default_value = "0.02")]
        distance_penalty: f32,

        /// Window size for WeDLM streaming parallel decoding
        #[arg(short, long, default_value = "32")]
        block_size: usize,

        /// Max tokens to accept per iteration within a block
        #[arg(long, default_value = "4")]
        max_per_step: usize,

        /// Use simple autoregressive decoding instead of WeDLM
        #[arg(long)]
        autoregressive: bool,
    },

    /// Test model loading
    Test {
        /// Model path or HuggingFace ID (e.g., tencent/WeDLM-8B-Instruct)
        #[arg(short, long)]
        model: String,
    },

    /// Benchmark autoregressive vs WeDLM parallel decoding
    Benchmark {
        /// Model path or HuggingFace ID (e.g., tencent/WeDLM-8B-Instruct)
        #[arg(short, long)]
        model: String,

        /// Number of tokens to generate per run
        #[arg(short = 'n', long, default_value = "64")]
        tokens: usize,

        /// Number of warmup runs
        #[arg(long, default_value = "1")]
        warmup: usize,

        /// Number of benchmark runs
        #[arg(long, default_value = "3")]
        runs: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.3")]
        temperature: f32,

        /// Window size for WeDLM streaming parallel decoding
        #[arg(short, long, default_value = "96")]
        block_size: usize,

        /// Entropy threshold for WeDLM decoding (lower = better quality, higher = faster)
        #[arg(short = 'e', long, default_value = "0.6")]
        entropy_threshold: f32,

        /// Margin threshold: require logit(top1) - logit(top2) >= this value
        #[arg(long, default_value = "0.0")]
        margin_threshold: f32,

        /// Distance penalty coefficient λ for H̃ = H + λ·d selection
        #[arg(long, default_value = "0.02")]
        distance_penalty: f32,

        /// Max tokens to accept per iteration within a block
        #[arg(long, default_value = "4")]
        max_per_step: usize,
    },

    /// Sweep parameters to find optimal WeDLM configuration
    Sweep {
        /// Model path or HuggingFace ID (e.g., tencent/WeDLM-8B-Instruct)
        #[arg(short, long)]
        model: String,

        /// Number of tokens to generate per run
        #[arg(short = 'n', long, default_value = "64")]
        tokens: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.3")]
        temperature: f32,
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
            entropy_threshold,
            margin_threshold,
            distance_penalty,
            block_size,
            max_per_step,
            autoregressive,
        } => {
            // Resolve model path (local or HuggingFace)
            let model_path = resolve_model_path(&model)?;

            // Model loading with timing
            eprintln!("Model loading...");
            let load_start = Instant::now();
            let engine = WeDLMEngine::from_pretrained(&model_path)?;
            let load_time = load_start.elapsed();
            eprintln!("Model loaded in {:.2}s", load_time.as_secs_f64());

            // Generation with timing
            eprintln!("\nGenerating...");
            let gen_start = Instant::now();
            let output = if autoregressive {
                engine.generate_autoregressive(&prompt, max_tokens, temperature)?
            } else {
                let params = SamplingParams {
                    temperature,
                    entropy_threshold,
                    margin_threshold,
                    distance_penalty,
                    max_tokens_per_step: max_per_step,
                    ..Default::default()
                };
                engine.generate_with_block_size(&prompt, max_tokens, block_size, Some(params))?
            };
            let gen_time = gen_start.elapsed();

            // Count output tokens (approximate by splitting on whitespace + punctuation)
            let prompt_len = prompt.len();
            let output_only = &output[prompt_len..].trim();
            let token_count = output_only.split_whitespace().count().max(1);
            let tokens_per_sec = token_count as f64 / gen_time.as_secs_f64();

            // Output
            eprintln!("\n--- Output ---");
            println!("{}", output);
            
            // Stats
            eprintln!("\n--- Stats ---");
            eprintln!("TTS (time to generate): {:.2}s", gen_time.as_secs_f64());
            eprintln!("Tokens generated: ~{}", token_count);
            eprintln!("Speed: {:.1} tok/s", tokens_per_sec);
        }

        Commands::Test { model } => {
            let model_path = resolve_model_path(&model)?;
            tracing::info!("Testing model loading from {:?}...", model_path);

            // Load config
            let config_path = model_path.join("config.json");
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
            let vb = wedlm_rs::weights::load_model_vb(&model_path, dtype, &device)?;
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

        Commands::Benchmark {
            model,
            tokens,
            warmup,
            runs,
            temperature,
            block_size,
            entropy_threshold,
            margin_threshold,
            distance_penalty,
            max_per_step,
        } => {
            let model_path = resolve_model_path(&model)?;
            tracing::info!("Loading model from {:?}...", model_path);
            let engine = WeDLMEngine::from_pretrained(&model_path)?;

            let prompt = "The quick brown fox jumps over the lazy dog. In a world where technology";
            let params = SamplingParams {
                temperature,
                entropy_threshold,
                margin_threshold,
                distance_penalty,
                max_tokens_per_step: max_per_step,
                ..Default::default()
            };

            println!("\n=== WeDLM Benchmark ===");
            println!("Prompt: \"{}...\"", &prompt[..50.min(prompt.len())]);
            println!("Tokens per run: {}", tokens);
            println!("Block size: {}, Entropy thresh: {}, Margin: {}, Max/step: {}", block_size, entropy_threshold, margin_threshold, max_per_step);
            println!("Warmup runs: {}, Benchmark runs: {}", warmup, runs);
            println!();

            // Warmup for autoregressive
            println!("Warming up autoregressive...");
            for _ in 0..warmup {
                let _ = engine.generate_autoregressive(prompt, tokens, temperature)?;
            }

            // Benchmark autoregressive
            println!("Benchmarking autoregressive...");
            let mut ar_times = Vec::with_capacity(runs);
            for i in 0..runs {
                let start = Instant::now();
                let _output = engine.generate_autoregressive(prompt, tokens, temperature)?;
                let elapsed = start.elapsed();
                ar_times.push(elapsed);
                println!(
                    "  Run {}: {:.2}s ({:.1} tok/s)",
                    i + 1,
                    elapsed.as_secs_f64(),
                    tokens as f64 / elapsed.as_secs_f64()
                );
            }

            // Warmup for WeDLM
            println!("\nWarming up WeDLM parallel...");
            for _ in 0..warmup {
                let _ = engine.generate_with_block_size(prompt, tokens, block_size, Some(params.clone()))?;
            }

            // Benchmark WeDLM
            println!("Benchmarking WeDLM parallel...");
            let mut wedlm_times = Vec::with_capacity(runs);
            for i in 0..runs {
                let start = Instant::now();
                let _output = engine.generate_with_block_size(prompt, tokens, block_size, Some(params.clone()))?;
                let elapsed = start.elapsed();
                wedlm_times.push(elapsed);

                println!(
                    "  Run {}: {:.2}s ({:.1} tok/s)",
                    i + 1,
                    elapsed.as_secs_f64(),
                    tokens as f64 / elapsed.as_secs_f64()
                );
            }

            // Calculate averages
            let ar_avg: f64 = ar_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / runs as f64;
            let wedlm_avg: f64 = wedlm_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / runs as f64;

            let ar_tok_per_sec = tokens as f64 / ar_avg;
            let wedlm_tok_per_sec = tokens as f64 / wedlm_avg;
            let speedup = ar_avg / wedlm_avg;

            println!("\n=== Results ===");
            println!(
                "Autoregressive:  {:.2}s avg ({:.1} tok/s)",
                ar_avg, ar_tok_per_sec
            );
            println!(
                "WeDLM Parallel:  {:.2}s avg ({:.1} tok/s)",
                wedlm_avg, wedlm_tok_per_sec
            );
            println!();
            if speedup > 1.0 {
                println!("WeDLM is {:.2}x FASTER than autoregressive", speedup);
            } else {
                println!("WeDLM is {:.2}x SLOWER than autoregressive", 1.0 / speedup);
            }
        }

        Commands::Sweep {
            model,
            tokens,
            temperature,
        } => {
            let model_path = resolve_model_path(&model)?;
            tracing::info!("Loading model from {:?}...", model_path);
            let engine = WeDLMEngine::from_pretrained(&model_path)?;

            let prompt = "The quick brown fox jumps over the lazy dog. In a world where technology";

            // Parameter ranges to sweep
            let block_sizes = [8, 16, 32, 64];
            let entropy_thresholds = [1.0, 2.0, 3.0, 4.0, 6.0];
            let max_per_steps = [8, 16, 32, 64];

            println!("\n=== WeDLM Parameter Sweep ===");
            println!("Tokens: {}, Temperature: {}", tokens, temperature);
            println!("Testing {} configurations...\n", block_sizes.len() * entropy_thresholds.len() * max_per_steps.len());

            // First get autoregressive baseline
            println!("Getting autoregressive baseline...");
            let ar_start = Instant::now();
            let _output = engine.generate_autoregressive(prompt, tokens, temperature)?;
            let ar_time = ar_start.elapsed().as_secs_f64();
            let ar_tok_per_sec = tokens as f64 / ar_time;
            println!("Autoregressive: {:.2}s ({:.1} tok/s)\n", ar_time, ar_tok_per_sec);

            println!("{:>6} {:>8} {:>8} {:>8} {:>8}", "block", "entropy", "max/step", "tok/s", "speedup");
            println!("{}", "-".repeat(48));

            let mut best_speedup = 0.0f64;
            let mut best_config = (0usize, 0.0f32, 0usize);

            for &block_size in &block_sizes {
                for &entropy_threshold in &entropy_thresholds {
                    for &max_per_step in &max_per_steps {
                        // Skip configs where max_per_step > block_size (pointless)
                        if max_per_step > block_size {
                            continue;
                        }

                        let params = SamplingParams {
                            temperature,
                            entropy_threshold,
                            max_tokens_per_step: max_per_step,
                            ..Default::default()
                        };

                        let start = Instant::now();
                        let _output = engine.generate_with_block_size(
                            prompt, tokens, block_size, Some(params)
                        )?;
                        let elapsed = start.elapsed().as_secs_f64();
                        let tok_per_sec = tokens as f64 / elapsed;
                        let speedup = ar_time / elapsed;

                        println!(
                            "{:>6} {:>8.2} {:>8} {:>8.1} {:>7.2}x",
                            block_size, entropy_threshold, max_per_step, tok_per_sec, speedup
                        );

                        if speedup > best_speedup {
                            best_speedup = speedup;
                            best_config = (block_size, entropy_threshold, max_per_step);
                        }
                    }
                }
            }

            println!("\n=== Best Configuration ===");
            println!(
                "block_size={}, entropy_threshold={:.2}, max_per_step={} -> {:.2}x speedup",
                best_config.0, best_config.1, best_config.2, best_speedup
            );
        }
    }

    Ok(())
}
