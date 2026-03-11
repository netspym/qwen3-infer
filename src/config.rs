//! Configuration module for Qwen3 inference
//!
//! Loads settings from config.toml file

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Root configuration structure
#[derive(Debug, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub generation: GenerationConfig,
    pub prompt: PromptConfig,
}

/// Model configuration
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Model name on ModelScope (e.g., "Qwen/Qwen3-0.6B")
    /// For local models, use "local:./path/to/model"
    pub name: String,

    /// Model type: "chat" or "embedding"
    #[serde(rename = "type")]
    pub model_type: ModelType,

    /// Model source: "modelscope.cn" or "local"
    #[serde(default = "default_source")]
    pub source: String,

    /// Data type: "f32" or "f16"
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Architecture: "dense" (default) or "moe"
    #[serde(default = "default_arch")]
    pub arch: String,
}

fn default_arch() -> String { "dense".to_string() }

/// Model type enum
#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Chat,
    Embedding,
}

/// Generation parameters
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: f64,

    /// Random seed
    #[serde(default = "default_seed")]
    pub seed: u64,
}

/// Prompt configuration
#[derive(Debug, Deserialize)]
pub struct PromptConfig {
    /// System prompt (optional)
    #[serde(default)]
    pub system: String,

    /// User message
    pub user: String,

    /// Disable thinking mode
    #[serde(default)]
    pub no_think: bool,

    /// Chat template format
    #[serde(default = "default_template")]
    pub template: String,
}

// Default value functions
fn default_source() -> String { "modelscope.cn".to_string() }
fn default_dtype() -> String { "f32".to_string() }
fn default_max_tokens() -> usize { 500 }
fn default_temperature() -> f64 { 0.7 }
fn default_top_p() -> f64 { 0.9 }
fn default_seed() -> u64 { 42 }
fn default_template() -> String { "chatml".to_string() }

impl Config {
    /// Load configuration from a TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
    }

    /// Load from default config.toml in current directory
    pub fn load_default() -> Result<Self> {
        Self::load(Path::new("config.toml"))
    }
}

impl ModelConfig {
    /// Check if this is a local model
    pub fn is_local(&self) -> bool {
        self.source == "local" || self.name.starts_with("local:")
    }

    /// Get the local path for a local model
    pub fn local_path(&self) -> Option<std::path::PathBuf> {
        if self.name.starts_with("local:") {
            Some(std::path::PathBuf::from(&self.name[6..]))
        } else if self.source == "local" {
            Some(std::path::PathBuf::from(&self.name))
        } else {
            None
        }
    }

    /// Get the cache directory name for this model
    pub fn cache_dir_name(&self) -> String {
        if let Some(local_path) = self.local_path() {
            // For local models, use the directory name
            local_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "local_model".to_string())
        } else {
            // Convert "Qwen/Qwen3-0.6B" to "Qwen__Qwen3-0.6B"
            self.name.replace('/', "__")
        }
    }

    /// Get the ModelScope model ID
    pub fn model_id(&self) -> &str {
        &self.name
    }

    /// Check if this is an embedding model
    pub fn is_embedding(&self) -> bool {
        self.model_type == ModelType::Embedding
    }

    /// Check if this is a MoE model
    pub fn is_moe(&self) -> bool {
        self.arch == "moe" || self.name.contains("30B-A3B") || self.name.contains("30b-a3b")
    }

    /// Get the download URL base for ModelScope
    pub fn download_url(&self, filename: &str) -> String {
        format!(
            "https://{}/api/v1/models/{}/repo?Revision=master&FilePath={}",
            self.source, self.name, filename
        )
    }
}

impl PromptConfig {
    /// Build the full prompt using the chat template
    pub fn build_prompt(&self) -> String {
        let mut prompt = String::new();

        // Add system prompt if present
        if !self.system.is_empty() {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(&self.system);
            prompt.push_str("<|im_end|>\n");
        }

        // Add user message
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(&self.user);

        // Add no_think tag if enabled
        if self.no_think {
            prompt.push_str("/no_think");
        }

        prompt.push_str("<|im_end|>\n");
        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Chat => write!(f, "chat"),
            ModelType::Embedding => write!(f, "embedding"),
        }
    }
}
