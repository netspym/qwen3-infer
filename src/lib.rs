//! Qwen3 Inference Python Library
//!
//! Provides Python bindings for Qwen3 models (chat and embedding).
//!
//! Usage:
//! ```python
//! from qwen3_infer import Qwen3Model
//!
//! model = Qwen3Model("Qwen/Qwen3-0.6B", model_dir="./model")
//! response = model.generate("你好", max_tokens=100)
//! ```

pub mod config;
pub mod qwen3;
pub mod qwen3_moe;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;

use qwen3::{Qwen3Config, Qwen3ForCausalLM};

/// Download a file from ModelScope
fn download_file(url: &str, file_path: &PathBuf) -> Result<(), String> {
    if file_path.exists() {
        return Ok(());
    }

    println!("  Downloading: {}", file_path.file_name().unwrap().to_string_lossy());

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let response = client
        .get(url)
        .header("User-Agent", "qwen3-infer/0.1.0")
        .send()
        .map_err(|e| format!("Failed to download: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed: HTTP {}", response.status()));
    }

    let content = response.bytes()
        .map_err(|e| format!("Failed to read response: {}", e))?;

    let mut file = fs::File::create(file_path)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    file.write_all(&content)
        .map_err(|e| format!("Failed to write file: {}", e))?;

    println!("    Downloaded {} bytes", content.len());
    Ok(())
}

/// Get ModelScope download URL
fn get_modelscope_url(model_name: &str, filename: &str) -> String {
    format!(
        "https://modelscope.cn/api/v1/models/{}/repo?Revision=master&FilePath={}",
        model_name, filename
    )
}

/// Qwen3 Model wrapper for Python
///
/// Holds the model in memory for fast inference.
#[pyclass]
pub struct Qwen3Model {
    model: Mutex<Qwen3ForCausalLM>,
    tokenizer: Tokenizer,
    device: Device,
    model_type: String,
}

#[pymethods]
impl Qwen3Model {
    /// Load a Qwen3 model
    ///
    /// Args:
    ///     model_name: Model name on ModelScope (e.g., "Qwen/Qwen3-0.6B")
    ///     model_dir: Directory to store/load model files (default: "./model")
    ///     model_type: "chat" or "embedding" (default: "chat")
    ///
    /// Returns:
    ///     Qwen3Model instance with model loaded in memory
    #[new]
    #[pyo3(signature = (model_name, model_dir="./model", model_type="chat"))]
    fn new(model_name: &str, model_dir: &str, model_type: &str) -> PyResult<Self> {
        println!("Loading Qwen3 model: {}", model_name);
        println!("Model directory: {}", model_dir);
        println!("Model type: {}", model_type);

        // Create model directory
        let model_path = PathBuf::from(model_dir);
        fs::create_dir_all(&model_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create model directory: {}", e)))?;

        // Download model files
        println!("Checking model files...");
        let tokenizer_path = model_path.join("tokenizer.json");
        let config_path = model_path.join("config.json");
        let weights_path = model_path.join("model.safetensors");

        download_file(&get_modelscope_url(model_name, "tokenizer.json"), &tokenizer_path)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        download_file(&get_modelscope_url(model_name, "config.json"), &config_path)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        download_file(&get_modelscope_url(model_name, "model.safetensors"), &weights_path)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        // Load tokenizer
        println!("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e)))?;

        // Load config
        println!("Loading config...");
        let config: Qwen3Config = qwen3::load_config(&config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load config: {}", e)))?;

        println!("  {} layers, {} heads, head_dim={}",
                 config.num_hidden_layers, config.num_attention_heads, config.head_dim);

        // Setup device
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Load weights
        println!("Loading model weights...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to load weights: {}", e)))?
        };

        // Build model
        println!("Building model...");
        let model = Qwen3ForCausalLM::new(&config, vb)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build model: {}", e)))?;

        println!("Model loaded successfully!");

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            model_type: model_type.to_string(),
        })
    }

    /// Generate text (for chat models)
    ///
    /// Args:
    ///     prompt: User message
    ///     max_tokens: Maximum tokens to generate (default: 500)
    ///     temperature: Sampling temperature (default: 0.7)
    ///     top_p: Top-p sampling (default: 0.9)
    ///     system: Optional system prompt
    ///     no_think: Disable thinking mode (default: True)
    ///
    /// Returns:
    ///     Generated text
    #[pyo3(signature = (prompt, max_tokens=500, temperature=0.7, top_p=0.9, system="", no_think=true))]
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        system: &str,
        no_think: bool,
    ) -> PyResult<String> {
        if self.model_type != "chat" {
            return Err(PyRuntimeError::new_err("generate() is only for chat models. Use embed() for embedding models."));
        }

        // Build chat prompt
        let full_prompt = build_chat_prompt(prompt, system, no_think);

        // Tokenize
        let encoding = self.tokenizer
            .encode(full_prompt.as_str(), true)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let input_tensor = Tensor::new(&tokens[..], &self.device)
            .map_err(|e| PyRuntimeError::new_err(format!("Tensor error: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| PyRuntimeError::new_err(format!("Tensor error: {}", e)))?;

        // Get model lock
        let mut model = self.model.lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        model.clear_cache();

        // Logits processor for sampling
        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));

        // First forward pass
        let logits = model.forward(&input_tensor, 0)
            .map_err(|e| PyRuntimeError::new_err(format!("Forward error: {}", e)))?;

        let seq_len = logits.dim(1)
            .map_err(|e| PyRuntimeError::new_err(format!("Dim error: {}", e)))?;

        let next_token_logits = logits
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| PyRuntimeError::new_err(format!("Narrow error: {}", e)))?
            .squeeze(1)
            .map_err(|e| PyRuntimeError::new_err(format!("Squeeze error: {}", e)))?
            .squeeze(0)
            .map_err(|e| PyRuntimeError::new_err(format!("Squeeze error: {}", e)))?;

        let mut next_token = logits_processor.sample(&next_token_logits)
            .map_err(|e| PyRuntimeError::new_err(format!("Sample error: {}", e)))?;

        let mut generated_tokens = Vec::new();
        let eos_tokens = [151643u32, 151644, 151645];

        if !eos_tokens.contains(&next_token) {
            generated_tokens.push(next_token);
            tokens.push(next_token);
        }

        // Generation loop
        for _ in 1..max_tokens {
            if eos_tokens.contains(&next_token) {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| PyRuntimeError::new_err(format!("Tensor error: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| PyRuntimeError::new_err(format!("Tensor error: {}", e)))?;

            let logits = model.forward(&input, tokens.len() - 1)
                .map_err(|e| PyRuntimeError::new_err(format!("Forward error: {}", e)))?;

            let next_token_logits = logits
                .squeeze(1)
                .map_err(|e| PyRuntimeError::new_err(format!("Squeeze error: {}", e)))?
                .squeeze(0)
                .map_err(|e| PyRuntimeError::new_err(format!("Squeeze error: {}", e)))?;

            next_token = logits_processor.sample(&next_token_logits)
                .map_err(|e| PyRuntimeError::new_err(format!("Sample error: {}", e)))?;

            if eos_tokens.contains(&next_token) {
                break;
            }

            generated_tokens.push(next_token);
            tokens.push(next_token);
        }

        // Decode generated tokens
        let generated_text = self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| PyRuntimeError::new_err(format!("Decode error: {}", e)))?;

        Ok(generated_text)
    }

    /// Generate embedding (for embedding models)
    ///
    /// Args:
    ///     text: Text to embed
    ///
    /// Returns:
    ///     List of floats (embedding vector)
    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        if self.model_type != "embedding" {
            return Err(PyRuntimeError::new_err("embed() is only for embedding models. Use generate() for chat models."));
        }

        // Tokenize
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| PyRuntimeError::new_err(format!("Tokenization failed: {}", e)))?;

        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        let num_tokens = tokens.len();

        let input_tensor = Tensor::new(&tokens[..], &self.device)
            .map_err(|e| PyRuntimeError::new_err(format!("Tensor error: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| PyRuntimeError::new_err(format!("Tensor error: {}", e)))?;

        // Get model lock
        let mut model = self.model.lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        model.clear_cache();

        // Get hidden states
        let hidden_states = model.forward_hidden(&input_tensor)
            .map_err(|e| PyRuntimeError::new_err(format!("Forward error: {}", e)))?;

        // Mean pooling
        let embedding = hidden_states.sum(1)
            .map_err(|e| PyRuntimeError::new_err(format!("Sum error: {}", e)))?;

        let embedding = (embedding / num_tokens as f64)
            .map_err(|e| PyRuntimeError::new_err(format!("Div error: {}", e)))?;

        let embedding = embedding.squeeze(0)
            .map_err(|e| PyRuntimeError::new_err(format!("Squeeze error: {}", e)))?;

        // L2 normalize
        let norm = embedding.sqr()
            .map_err(|e| PyRuntimeError::new_err(format!("Sqr error: {}", e)))?
            .sum_all()
            .map_err(|e| PyRuntimeError::new_err(format!("Sum error: {}", e)))?
            .sqrt()
            .map_err(|e| PyRuntimeError::new_err(format!("Sqrt error: {}", e)))?;

        let embedding = embedding.broadcast_div(&norm)
            .map_err(|e| PyRuntimeError::new_err(format!("Div error: {}", e)))?;

        let result: Vec<f32> = embedding.to_vec1()
            .map_err(|e| PyRuntimeError::new_err(format!("ToVec error: {}", e)))?;

        Ok(result)
    }

    /// Get model info
    fn info(&self) -> PyResult<String> {
        Ok(format!("Qwen3Model(type={})", self.model_type))
    }
}

/// Build chat prompt with ChatML format
fn build_chat_prompt(user_msg: &str, system: &str, no_think: bool) -> String {
    let mut prompt = String::new();

    if !system.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user_msg);

    if no_think {
        prompt.push_str("/no_think");
    }

    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");

    prompt
}

/// Python module
#[pymodule]
fn qwen3_infer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Qwen3Model>()?;
    Ok(())
}
