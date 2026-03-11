//! Qwen3 Inference - Configurable Model Runner
//!
//! Supports chat and embedding models from ModelScope or local paths.
//! Supports both dense and MoE (Mixture of Experts) architectures.
//! Configuration is loaded from config.toml.

mod config;
mod qwen3;
mod qwen3_moe;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

use config::{Config, ModelType};
use qwen3::{Qwen3Config, Qwen3ForCausalLM};
use qwen3_moe::{Qwen3MoEConfig, Qwen3MoEForCausalLM};

// ============================================================================
// Model Wrapper (supports both dense and MoE)
// ============================================================================

enum ModelWrapper {
    Dense(Qwen3ForCausalLM),
    MoE(Qwen3MoEForCausalLM),
}

impl ModelWrapper {
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            ModelWrapper::Dense(m) => m.forward(input_ids, start_pos),
            ModelWrapper::MoE(m) => m.forward(input_ids, start_pos),
        }
    }

    fn forward_hidden(&mut self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            ModelWrapper::Dense(m) => m.forward_hidden(input_ids),
            ModelWrapper::MoE(m) => m.forward_hidden(input_ids),
        }
    }

    fn clear_cache(&mut self) {
        match self {
            ModelWrapper::Dense(m) => m.clear_cache(),
            ModelWrapper::MoE(m) => m.clear_cache(),
        }
    }
}

// ============================================================================
// Model Download
// ============================================================================

/// Download a file from ModelScope with progress bar
fn download_file(url: &str, filename: &str, cache_dir: &PathBuf) -> Result<PathBuf> {
    let file_path = cache_dir.join(filename);

    if file_path.exists() {
        println!("  [cached] {}", filename);
        return Ok(file_path);
    }

    println!("  [downloading] {}...", filename);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    let response = client
        .get(url)
        .header("User-Agent", "qwen3-infer/0.1.0")
        .send()
        .context("Failed to connect to model source")?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", response.status());
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    {spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let content = response.bytes()?;
    pb.finish_and_clear();

    let mut file = fs::File::create(&file_path)?;
    file.write_all(&content)?;

    println!("    Downloaded {} bytes", content.len());

    Ok(file_path)
}

// ============================================================================
// Text Generation
// ============================================================================

struct GenerationStats {
    prompt_tokens: usize,
    generated_tokens: usize,
    prompt_time_ms: u128,
    generation_time_ms: u128,
}

impl GenerationStats {
    fn print(&self) {
        println!("\n--- Performance ---");
        println!("  Prompt tokens:     {}", self.prompt_tokens);
        println!("  Generated tokens:  {}", self.generated_tokens);

        if self.prompt_time_ms > 0 {
            let tps = (self.prompt_tokens as f64 * 1000.0) / self.prompt_time_ms as f64;
            println!("  Prompt processing: {:.1} tokens/sec ({} ms)", tps, self.prompt_time_ms);
        }

        if self.generation_time_ms > 0 && self.generated_tokens > 0 {
            let tps = (self.generated_tokens as f64 * 1000.0) / self.generation_time_ms as f64;
            println!("  Generation speed:  {:.1} tokens/sec ({} ms)", tps, self.generation_time_ms);
        }
    }
}

/// Generate text using the chat model
fn generate_text(
    model: &mut ModelWrapper,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &Config,
    device: &Device,
) -> Result<GenerationStats> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = tokens.len();

    let input_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

    let mut logits_processor = LogitsProcessor::new(
        config.generation.seed,
        Some(config.generation.temperature),
        Some(config.generation.top_p),
    );

    model.clear_cache();

    // Prompt processing
    let prompt_start = Instant::now();
    let logits = model.forward(&input_tensor, 0)?;
    let prompt_time_ms = prompt_start.elapsed().as_millis();

    let seq_len = logits.dim(1)?;
    let next_token_logits = logits
        .narrow(1, seq_len - 1, 1)?
        .squeeze(1)?
        .squeeze(0)?;

    let mut next_token = logits_processor.sample(&next_token_logits)?;

    let mut generated_tokens = Vec::new();
    let eos_tokens = [151643u32, 151644, 151645];

    // Generation loop
    let generation_start = Instant::now();

    if !eos_tokens.contains(&next_token) {
        generated_tokens.push(next_token);
        tokens.push(next_token);

        if let Ok(text) = tokenizer.decode(&[next_token], true) {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }

    for _ in 1..config.generation.max_tokens {
        if eos_tokens.contains(&next_token) {
            break;
        }

        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() - 1)?;

        let next_token_logits = logits.squeeze(1)?.squeeze(0)?;
        next_token = logits_processor.sample(&next_token_logits)?;

        if eos_tokens.contains(&next_token) {
            break;
        }

        generated_tokens.push(next_token);
        tokens.push(next_token);

        if let Ok(text) = tokenizer.decode(&[next_token], true) {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }

    let generation_time_ms = generation_start.elapsed().as_millis();

    Ok(GenerationStats {
        prompt_tokens: prompt_len,
        generated_tokens: generated_tokens.len(),
        prompt_time_ms,
        generation_time_ms,
    })
}

/// Generate embeddings using the embedding model
fn generate_embedding(
    model: &mut ModelWrapper,
    tokenizer: &Tokenizer,
    text: &str,
    device: &Device,
) -> Result<Vec<f32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    let num_tokens = tokens.len();
    let input_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

    model.clear_cache();

    // Get hidden states from the model
    let hidden_states = model.forward_hidden(&input_tensor)?;

    // Mean pooling over sequence dimension: [batch, seq, hidden] -> [batch, hidden]
    let embedding = hidden_states.sum(1)?;
    let embedding = (embedding / num_tokens as f64)?;
    let embedding = embedding.squeeze(0)?;

    // L2 normalize
    let norm = embedding.sqr()?.sum_all()?.sqrt()?;
    let embedding = embedding.broadcast_div(&norm)?;

    Ok(embedding.to_vec1()?)
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("=== Qwen3 Inference ===\n");

    // Load configuration
    let config = Config::load_default()
        .context("Failed to load config.toml. Make sure the file exists in the current directory.")?;

    let is_local = config.model.is_local();
    let is_moe = config.model.is_moe();

    println!("Configuration loaded from config.toml");
    println!("  Model:       {}", config.model.name);
    println!("  Type:        {}", config.model.model_type);
    println!("  Source:      {}", if is_local { "local" } else { &config.model.source });
    println!("  Architecture:{}", if is_moe { "MoE" } else { "Dense" });

    if config.model.model_type == ModelType::Chat {
        println!("  Max tokens:  {}", config.generation.max_tokens);
        println!("  Temperature: {}", config.generation.temperature);
        println!("  Top-p:       {}", config.generation.top_p);
        println!("  No-think:    {}", config.prompt.no_think);
    }
    println!();

    // Configure thread pool
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .ok();

    println!("CPU threads: {}", num_cpus);

    // Print BLAS backend info
    #[cfg(feature = "accelerate")]
    println!("BLAS backend: Apple Accelerate");
    #[cfg(feature = "mkl")]
    println!("BLAS backend: Intel MKL");
    #[cfg(not(any(feature = "accelerate", feature = "mkl")))]
    println!("BLAS backend: None (consider using --features accelerate or mkl)");

    println!();

    // Setup device and dtype
    let device = Device::Cpu;
    let dtype = match config.model.dtype.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        _ => DType::F32,
    };

    // Get model files (local or download)
    let (tokenizer_path, config_path, model_path) = if is_local {
        let local_path = config.model.local_path()
            .ok_or_else(|| anyhow::anyhow!("Invalid local model path"))?;

        println!("Loading from local path: {:?}\n", local_path);

        let tokenizer_path = local_path.join("tokenizer.json");
        let config_path = local_path.join("config.json");
        let model_path = local_path.join("model.safetensors");

        if !model_path.exists() {
            anyhow::bail!("Model file not found: {:?}", model_path);
        }

        (tokenizer_path, config_path, model_path)
    } else {
        // Setup cache directory for remote models
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("modelscope")
            .join(config.model.cache_dir_name());
        fs::create_dir_all(&cache_dir)?;

        println!("Cache: {:?}\n", cache_dir);

        // Download model files
        println!("Step 1: Downloading model files...");
        let tokenizer_url = config.model.download_url("tokenizer.json");
        let config_url = config.model.download_url("config.json");
        let model_url = config.model.download_url("model.safetensors");

        let tokenizer_path = download_file(&tokenizer_url, "tokenizer.json", &cache_dir)?;
        let config_path = download_file(&config_url, "config.json", &cache_dir)?;
        let model_path = download_file(&model_url, "model.safetensors", &cache_dir)?;
        println!();

        (tokenizer_path, config_path, model_path)
    };

    // Load tokenizer
    println!("Step 2: Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!();

    // Load model weights
    println!("Step 3: Loading model weights...");
    let load_start = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
    println!("  Loaded in {} ms", load_start.elapsed().as_millis());
    println!();

    // Build model based on architecture
    println!("Step 4: Building model...");
    let build_start = Instant::now();

    let mut model: ModelWrapper = if is_moe {
        // Load MoE model
        println!("  Loading MoE configuration...");
        let model_config: Qwen3MoEConfig = qwen3_moe::load_moe_config(&config_path)?;
        println!("  {} layers, {} heads, {} experts, {} active per token",
                 model_config.num_hidden_layers,
                 model_config.num_attention_heads,
                 model_config.num_experts,
                 model_config.num_experts_per_tok);

        ModelWrapper::MoE(Qwen3MoEForCausalLM::new(&model_config, vb)?)
    } else {
        // Load dense model
        println!("  Loading dense configuration...");
        let model_config: Qwen3Config = qwen3::load_config(&config_path)?;
        println!("  {} layers, {} heads, head_dim={}",
                 model_config.num_hidden_layers,
                 model_config.num_attention_heads,
                 model_config.head_dim);

        ModelWrapper::Dense(Qwen3ForCausalLM::new(&model_config, vb)?)
    };

    println!("  Built in {} ms", build_start.elapsed().as_millis());
    println!();

    // Run inference based on model type
    match config.model.model_type {
        ModelType::Chat => {
            println!("Step 5: Generating text...\n");

            let prompt = config.prompt.build_prompt();

            println!("--- Prompt ---");
            println!("{}", prompt);
            println!("--- Response ---");

            let stats = generate_text(&mut model, &tokenizer, &prompt, &config, &device)?;
            stats.print();
        }
        ModelType::Embedding => {
            println!("Step 5: Generating embedding...\n");

            println!("Input text: {}", config.prompt.user);

            let start = Instant::now();
            let embedding = generate_embedding(&mut model, &tokenizer, &config.prompt.user, &device)?;
            let elapsed = start.elapsed().as_millis();

            println!("\n--- Embedding ---");
            println!("  Dimension: {}", embedding.len());
            println!("  Time: {} ms", elapsed);
            println!("  First 10 values: {:?}", &embedding[..10.min(embedding.len())]);
        }
    }

    println!("\n=== Complete ===");

    Ok(())
}
