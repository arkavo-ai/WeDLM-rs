//! WeDLM-rs CLI
//!
//! Command-line interface for WeDLM inference.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use wedlm_rs::decoding::SamplingParams;
use wedlm_rs::WeDLMEngine;

// ============================================================================
// Quality Metrics
// ============================================================================

/// Quality metrics for detecting garbage output
#[derive(Debug, Clone)]
struct QualityMetrics {
    /// Longest run of consecutive identical tokens
    max_consecutive_repeats: usize,
    /// Ratio of repeated tokens in last 64 tokens (0.0 = all unique, 1.0 = all same)
    repeat_ratio_last_64: f32,
    /// Longest run of punctuation/whitespace characters
    max_punctuation_run: usize,
    /// Whether quality passes all thresholds
    passed: bool,
}

impl QualityMetrics {
    /// Thresholds for quality gate (tuned from sweep data)
    /// Good output: repeats=1-4, ratio=0.62-0.70, punct=1-4
    /// Bad output:  repeats=15-30, ratio=0.83-0.94, punct=16-31
    const MAX_REPEATS_THRESHOLD: usize = 8;
    const REPEAT_RATIO_THRESHOLD: f32 = 0.78;
    const MAX_PUNCT_THRESHOLD: usize = 10;

    /// Compute quality metrics from generated tokens and text
    fn compute(tokens: &[u32], text: &str) -> Self {
        // Max consecutive repeats
        let mut max_repeat = 1;
        let mut current_repeat = 1;
        for i in 1..tokens.len() {
            if tokens[i] == tokens[i - 1] {
                current_repeat += 1;
                max_repeat = max_repeat.max(current_repeat);
            } else {
                current_repeat = 1;
            }
        }

        // Repeat ratio in last 64 tokens
        let last_n = tokens.len().min(64);
        let last_tokens = &tokens[tokens.len().saturating_sub(last_n)..];
        let unique: HashSet<_> = last_tokens.iter().collect();
        let repeat_ratio = if last_tokens.is_empty() {
            0.0
        } else {
            1.0 - (unique.len() as f32 / last_tokens.len() as f32)
        };

        // Max punctuation/whitespace run
        let mut max_punct = 0;
        let mut current_punct = 0;
        for c in text.chars() {
            if c.is_ascii_punctuation() || c.is_whitespace() {
                current_punct += 1;
                max_punct = max_punct.max(current_punct);
            } else {
                current_punct = 0;
            }
        }

        let passed = max_repeat <= Self::MAX_REPEATS_THRESHOLD
            && repeat_ratio <= Self::REPEAT_RATIO_THRESHOLD
            && max_punct <= Self::MAX_PUNCT_THRESHOLD;

        Self {
            max_consecutive_repeats: max_repeat,
            repeat_ratio_last_64: repeat_ratio,
            max_punctuation_run: max_punct,
            passed,
        }
    }

    fn status_str(&self) -> &'static str {
        if self.passed { "PASS" } else { "FAIL" }
    }
}

// ============================================================================
// Presets
// ============================================================================

/// Preset configurations for quality/speed tradeoff
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum Preset {
    /// High quality, lower speed (max_per_step=2, entropy=0.4, λ=0.25)
    Quality,
    /// Balanced quality and speed (max_per_step=6, entropy=0.8, λ=0.15)
    #[default]
    Balanced,
    /// Maximum speed, lower quality (max_per_step=16, entropy=1.5, λ=0.05)
    Fast,
}

impl Preset {
    fn to_params(self, temperature: f32) -> SamplingParams {
        match self {
            Preset::Quality => SamplingParams {
                temperature,
                entropy_threshold: 0.4,
                distance_penalty: 0.25,
                max_tokens_per_step: 2,
                ..Default::default()
            },
            Preset::Balanced => SamplingParams {
                temperature,
                entropy_threshold: 0.8,
                distance_penalty: 0.15,
                max_tokens_per_step: 6,
                ..Default::default()
            },
            Preset::Fast => SamplingParams {
                temperature,
                entropy_threshold: 1.5,
                distance_penalty: 0.05,
                max_tokens_per_step: 16,
                ..Default::default()
            },
        }
    }

    fn default_block_size(self) -> usize {
        match self {
            Preset::Quality => 16,
            Preset::Balanced => 32,
            Preset::Fast => 64,
        }
    }
}

// ============================================================================
// CLI
// ============================================================================

/// Resolve a model path - either local directory or HuggingFace model ID
fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let local_path = PathBuf::from(model);

    if local_path.exists() {
        return Ok(local_path);
    }

    if !model.contains('/') {
        anyhow::bail!(
            "Model '{}' not found locally and doesn't look like a HF model ID (expected 'org/model')",
            model
        );
    }

    eprintln!("Resolving HuggingFace model: {}", model);

    let api = Api::new().context("Failed to initialize HuggingFace API")?;
    let repo = api.repo(Repo::new(model.to_string(), RepoType::Model));

    let config_path = repo
        .get("config.json")
        .context("Failed to fetch config.json - model may not exist or not be cached")?;

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

        /// Preset configuration (quality, balanced, fast)
        #[arg(long, value_enum, default_value = "balanced")]
        preset: Preset,

        /// Window size (overrides preset default)
        #[arg(short, long)]
        block_size: Option<usize>,

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

        /// Preset configuration (quality, balanced, fast)
        #[arg(long, value_enum, default_value = "balanced")]
        preset: Preset,

        /// Window size (overrides preset default)
        #[arg(short, long)]
        block_size: Option<usize>,

        /// Disable quality check
        #[arg(long)]
        no_quality_check: bool,
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

        /// Window size
        #[arg(short, long, default_value = "32")]
        block_size: usize,
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
            preset,
            block_size,
            autoregressive,
        } => {
            let model_path = resolve_model_path(&model)?;

            // Show warning for fast preset
            if matches!(preset, Preset::Fast) && !autoregressive {
                eprintln!("WARNING: Fast preset prioritizes speed over quality. Output may contain repetition.\n");
            }

            eprintln!("Model loading...");
            let load_start = Instant::now();
            let engine = WeDLMEngine::from_pretrained(&model_path)?;
            let load_time = load_start.elapsed();
            eprintln!("Model loaded in {:.2}s", load_time.as_secs_f64());

            eprintln!("\nGenerating...");
            let gen_start = Instant::now();
            let (output, token_count) = if autoregressive {
                engine.generate_autoregressive(&prompt, max_tokens, temperature)?
            } else {
                let params = preset.to_params(temperature);
                let bs = block_size.unwrap_or_else(|| preset.default_block_size());
                engine.generate_with_block_size(&prompt, max_tokens, bs, Some(params))?
            };
            let gen_time = gen_start.elapsed();

            let tokens_per_sec = token_count as f64 / gen_time.as_secs_f64();

            eprintln!("\n--- Output ---");
            println!("{}", output);

            eprintln!("\n--- Stats ---");
            eprintln!("Preset: {:?}", preset);
            eprintln!("TTS (time to generate): {:.2}s", gen_time.as_secs_f64());
            eprintln!("Tokens generated: {}", token_count);
            eprintln!("Speed: {:.1} tok/s", tokens_per_sec);
        }

        Commands::Test { model } => {
            let model_path = resolve_model_path(&model)?;
            tracing::info!("Testing model loading from {:?}...", model_path);

            let config_path = model_path.join("config.json");
            let config = wedlm_rs::WeDLMConfig::from_file(&config_path)?;
            tracing::info!(
                "Config loaded: {} layers, {} hidden",
                config.num_hidden_layers,
                config.hidden_size
            );

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

            tracing::info!("Building model...");
            let _model = wedlm_rs::model::WeDLMForCausalLM::new(&config, vb, &device)?;
            tracing::info!("Model built successfully!");

            tracing::info!("Testing forward pass...");
            let input = candle_core::Tensor::from_vec(vec![1i64, 2, 3], (1, 3), &device)?;
            let (logits, _) = _model.forward(&input, 0, None)?;
            tracing::info!("Forward pass output shape: {:?}", logits.dims());

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
            preset,
            block_size,
            no_quality_check,
        } => {
            let model_path = resolve_model_path(&model)?;
            tracing::info!("Loading model from {:?}...", model_path);
            let engine = WeDLMEngine::from_pretrained(&model_path)?;

            let prompt =
                "The quick brown fox jumps over the lazy dog. In a world where technology";
            let params = preset.to_params(temperature);
            let bs = block_size.unwrap_or_else(|| preset.default_block_size());

            println!("\n=== WeDLM Benchmark ===");
            println!("Prompt: \"{}...\"", &prompt[..50.min(prompt.len())]);
            println!("Tokens per run: {}", tokens);
            println!(
                "Preset: {:?}, Block size: {}, Entropy: {:.2}, λ: {:.2}, Max/step: {}",
                preset,
                bs,
                params.entropy_threshold,
                params.distance_penalty,
                params.max_tokens_per_step
            );
            println!("Warmup runs: {}, Benchmark runs: {}", warmup, runs);
            println!("Quality check: {}", if no_quality_check { "OFF" } else { "ON" });
            println!();

            // Warmup for autoregressive
            println!("Warming up autoregressive...");
            for _ in 0..warmup {
                let _ = engine.generate_autoregressive(prompt, tokens, temperature)?;
            }

            // Benchmark autoregressive
            println!("Benchmarking autoregressive...");
            let mut ar_times = Vec::with_capacity(runs);
            let mut ar_token_counts = Vec::with_capacity(runs);
            for i in 0..runs {
                let start = Instant::now();
                let (_output, token_count) = engine.generate_autoregressive(prompt, tokens, temperature)?;
                let elapsed = start.elapsed();
                ar_times.push(elapsed);
                ar_token_counts.push(token_count);
                println!(
                    "  Run {}: {:.2}s ({:.1} tok/s)",
                    i + 1,
                    elapsed.as_secs_f64(),
                    token_count as f64 / elapsed.as_secs_f64()
                );
            }

            // Warmup for WeDLM
            println!("\nWarming up WeDLM parallel...");
            for _ in 0..warmup {
                let _ = engine.generate_with_block_size(prompt, tokens, bs, Some(params.clone()))?;
            }

            // Benchmark WeDLM with quality metrics
            println!("Benchmarking WeDLM parallel...");
            let mut wedlm_times = Vec::with_capacity(runs);
            let mut wedlm_token_counts = Vec::with_capacity(runs);
            let mut quality_results: Vec<QualityMetrics> = Vec::with_capacity(runs);
            let mut last_output = String::new();

            for i in 0..runs {
                let start = Instant::now();
                let (output, token_count) =
                    engine.generate_with_block_size(prompt, tokens, bs, Some(params.clone()))?;
                let elapsed = start.elapsed();
                wedlm_times.push(elapsed);
                wedlm_token_counts.push(token_count);
                last_output = output.clone();

                // Compute quality metrics on generated output
                let output_tokens: Vec<u32> = output.chars().map(|c| c as u32).collect();
                let metrics = QualityMetrics::compute(&output_tokens, &output);
                let quality_str = if no_quality_check {
                    String::new()
                } else {
                    format!(
                        " [{}] repeats={}, ratio={:.2}, punct={}",
                        metrics.status_str(),
                        metrics.max_consecutive_repeats,
                        metrics.repeat_ratio_last_64,
                        metrics.max_punctuation_run
                    )
                };
                quality_results.push(metrics);

                println!(
                    "  Run {}: {:.2}s ({:.1} tok/s){}",
                    i + 1,
                    elapsed.as_secs_f64(),
                    token_count as f64 / elapsed.as_secs_f64(),
                    quality_str
                );
            }

            // Show sample output
            if !no_quality_check {
                println!("\n--- Sample Output (first 200 chars) ---");
                let sample_truncated: String = last_output.chars().take(200).collect();
                println!("{}", sample_truncated);
                if last_output.len() > 200 {
                    println!("...");
                }
            }

            // Calculate averages using actual token counts
            let ar_avg_time: f64 =
                ar_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / runs as f64;
            let wedlm_avg_time: f64 =
                wedlm_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / runs as f64;
            let ar_avg_tokens: f64 = ar_token_counts.iter().sum::<usize>() as f64 / runs as f64;
            let wedlm_avg_tokens: f64 = wedlm_token_counts.iter().sum::<usize>() as f64 / runs as f64;

            let ar_tok_per_sec = ar_avg_tokens / ar_avg_time;
            let wedlm_tok_per_sec = wedlm_avg_tokens / wedlm_avg_time;
            let speedup = ar_avg_time / wedlm_avg_time;

            // Quality summary
            let passed_count = quality_results.iter().filter(|m| m.passed).count();

            println!("\n=== Results ===");
            println!(
                "Autoregressive:  {:.2}s avg ({:.1} tok/s)",
                ar_avg_time, ar_tok_per_sec
            );
            println!(
                "WeDLM Parallel:  {:.2}s avg ({:.1} tok/s)",
                wedlm_avg_time, wedlm_tok_per_sec
            );

            if !no_quality_check {
                println!(
                    "Quality:         {}/{} runs passed",
                    passed_count, runs
                );
            }

            println!();
            if speedup > 1.0 {
                println!("WeDLM is {:.2}x FASTER than autoregressive", speedup);
            } else {
                println!(
                    "WeDLM is {:.2}x SLOWER than autoregressive",
                    1.0 / speedup
                );
            }

            // Final verdict
            if !no_quality_check && passed_count < runs {
                println!(
                    "\n⚠️  QUALITY GATE: {} of {} runs FAILED quality checks",
                    runs - passed_count,
                    runs
                );
                println!("   Consider using --preset quality for better output quality.");
            }
        }

        Commands::Sweep {
            model,
            tokens,
            temperature,
            block_size,
        } => {
            let model_path = resolve_model_path(&model)?;
            tracing::info!("Loading model from {:?}...", model_path);
            let engine = WeDLMEngine::from_pretrained(&model_path)?;

            let prompt =
                "The quick brown fox jumps over the lazy dog. In a world where technology";

            // Parameter ranges to sweep (with quality in mind)
            let entropy_thresholds = [0.5, 0.6, 0.7, 0.8, 1.0];
            let distance_penalties = [0.05, 0.10, 0.15, 0.20, 0.25];
            let max_per_steps = [2, 4, 6, 8];

            println!("\n=== WeDLM Parameter Sweep (Quality-Aware) ===");
            println!("Tokens: {}, Temperature: {}, Block size: {}", tokens, temperature, block_size);
            println!(
                "Testing {} configurations...\n",
                entropy_thresholds.len() * distance_penalties.len() * max_per_steps.len()
            );

            // First get autoregressive baseline
            println!("Getting autoregressive baseline...");
            let ar_start = Instant::now();
            let (_output, ar_token_count) = engine.generate_autoregressive(prompt, tokens, temperature)?;
            let ar_time = ar_start.elapsed().as_secs_f64();
            let ar_tok_per_sec = ar_token_count as f64 / ar_time;
            println!(
                "Autoregressive: {:.2}s ({:.1} tok/s)\n",
                ar_time, ar_tok_per_sec
            );

            println!(
                "{:>7} {:>6} {:>8} {:>7} {:>7} {:>7} {:>6}",
                "entropy", "lambda", "max/step", "tok/s", "speedup", "repeats", "status"
            );
            println!("{}", "-".repeat(60));

            #[derive(Debug)]
            struct SweepResult {
                entropy: f32,
                lambda: f32,
                max_per_step: usize,
                tok_per_sec: f64,
                speedup: f64,
                quality: QualityMetrics,
            }

            let mut results: Vec<SweepResult> = Vec::new();

            for &entropy_threshold in &entropy_thresholds {
                for &distance_penalty in &distance_penalties {
                    for &max_per_step in &max_per_steps {
                        let params = SamplingParams {
                            temperature,
                            entropy_threshold,
                            distance_penalty,
                            max_tokens_per_step: max_per_step,
                            ..Default::default()
                        };

                        let start = Instant::now();
                        let (output, token_count) = engine.generate_with_block_size(
                            prompt,
                            tokens,
                            block_size,
                            Some(params),
                        )?;
                        let elapsed = start.elapsed().as_secs_f64();
                        let tok_per_sec = token_count as f64 / elapsed;
                        let speedup = ar_time / elapsed;

                        // Quality check on generated output
                        let output_tokens: Vec<u32> =
                            output.chars().map(|c| c as u32).collect();
                        let quality = QualityMetrics::compute(&output_tokens, &output);

                        println!(
                            "{:>7.2} {:>6.2} {:>8} {:>7.1} {:>6.2}x {:>7} {:>6}",
                            entropy_threshold,
                            distance_penalty,
                            max_per_step,
                            tok_per_sec,
                            speedup,
                            quality.max_consecutive_repeats,
                            quality.status_str()
                        );

                        results.push(SweepResult {
                            entropy: entropy_threshold,
                            lambda: distance_penalty,
                            max_per_step,
                            tok_per_sec,
                            speedup,
                            quality,
                        });
                    }
                }
            }

            // Find Pareto frontier
            println!("\n=== Pareto Frontier ===");

            // Best quality (passing, lowest entropy)
            if let Some(best_quality) = results
                .iter()
                .filter(|r| r.quality.passed)
                .min_by(|a, b| a.entropy.partial_cmp(&b.entropy).unwrap())
            {
                println!(
                    "Best quality:   entropy={:.2}, λ={:.2}, max={} → {:.1} tok/s ({:.2}x)",
                    best_quality.entropy,
                    best_quality.lambda,
                    best_quality.max_per_step,
                    best_quality.tok_per_sec,
                    best_quality.speedup
                );
            }

            // Best balanced (passing, highest speed)
            if let Some(best_balanced) = results
                .iter()
                .filter(|r| r.quality.passed)
                .max_by(|a, b| a.tok_per_sec.partial_cmp(&b.tok_per_sec).unwrap())
            {
                println!(
                    "Best balanced:  entropy={:.2}, λ={:.2}, max={} → {:.1} tok/s ({:.2}x) PASS",
                    best_balanced.entropy,
                    best_balanced.lambda,
                    best_balanced.max_per_step,
                    best_balanced.tok_per_sec,
                    best_balanced.speedup
                );
            }

            // Best speed (any quality)
            if let Some(best_speed) = results
                .iter()
                .max_by(|a, b| a.tok_per_sec.partial_cmp(&b.tok_per_sec).unwrap())
            {
                println!(
                    "Best speed:     entropy={:.2}, λ={:.2}, max={} → {:.1} tok/s ({:.2}x) {}",
                    best_speed.entropy,
                    best_speed.lambda,
                    best_speed.max_per_step,
                    best_speed.tok_per_sec,
                    best_speed.speedup,
                    best_speed.quality.status_str()
                );
            }

            // Summary
            let pass_count = results.iter().filter(|r| r.quality.passed).count();
            println!(
                "\nQuality summary: {}/{} configurations passed quality gate",
                pass_count,
                results.len()
            );
        }
    }

    Ok(())
}
