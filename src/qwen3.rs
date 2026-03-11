//! Qwen3 Model Implementation for Candle (Optimized)
//!
//! This module implements the Qwen3 architecture with optimizations for CPU inference.
//! Key optimizations:
//! - Efficient KV cache management
//! - Minimized tensor allocations
//! - Contiguous memory layout where beneficial

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;

// ============================================================================
// Configuration
// ============================================================================

/// Qwen3 model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

// ============================================================================
// Rotary Position Embedding (RoPE) - Optimized
// ============================================================================

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Qwen3Config, dtype: DType, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim;
        let max_seq_len = config.max_position_embeddings;
        let theta = config.rope_theta;

        // Compute inverse frequencies
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;

        // Compute angles: [max_seq_len, half_dim]
        let angles = positions
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // Compute cos and sin
        let cos = angles.cos()?.to_dtype(dtype)?;
        let sin = angles.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    /// Apply rotary embedding using the rotate_half method
    #[inline]
    fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let head_dim = x.dim(3)?;
        let half_dim = head_dim / 2;

        // Get cos/sin for relevant positions: [seq_len, half_dim]
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        // Reshape for broadcasting: [1, 1, seq_len, half_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Split x into two halves
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&rotated_x1, &rotated_x2], D::Minus1)
    }
}

// ============================================================================
// Multi-Head Attention with GQA - Optimized
// ============================================================================

#[derive(Debug)]
struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    scale: f64,
}

impl Qwen3Attention {
    fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();

        Ok(Self {
            q_proj: linear_no_bias(hidden_size, q_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(hidden_size, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(hidden_size, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(q_dim, hidden_size, vb.pp("o_proj"))?,
            q_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            rotary_emb: RotaryEmbedding::new(config, vb.dtype(), vb.device())?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            kv_cache: None,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK normalization
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose: [batch, num_heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply rotary embedding
        let q = self.rotary_emb.apply(&q, start_pos)?;
        let k = self.rotary_emb.apply(&k, start_pos)?;

        // Update KV cache
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Expand KV for GQA
        let k = self.expand_kv(&k)?;
        let v = self.expand_kv(&v)?;

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;

        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    /// Expand KV heads for grouped query attention
    #[inline]
    fn expand_kv(&self, x: &Tensor) -> Result<Tensor> {
        let num_groups = self.num_heads / self.num_kv_heads;
        if num_groups == 1 {
            return Ok(x.clone());
        }

        let (batch, _, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((batch, self.num_kv_heads, num_groups, seq_len, head_dim))?
            .reshape((batch, self.num_heads, seq_len, head_dim))
    }

    #[inline]
    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ============================================================================
// MLP with SwiGLU
// ============================================================================

#[derive(Debug)]
struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MLP {
    fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let h = config.hidden_size;
        let i = config.intermediate_size;

        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    #[inline]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

#[derive(Debug)]
struct Qwen3Layer {
    input_layernorm: RmsNorm,
    self_attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Qwen3MLP,
}

impl Qwen3Layer {
    fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?,
            self_attn: Qwen3Attention::new(config, vb.pp("self_attn"))?,
            post_attention_layernorm: rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            mlp: Qwen3MLP::new(config, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        // Attention block with residual
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, start_pos)?;
        let x = (residual + x)?;

        // MLP block with residual
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    #[inline]
    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

// ============================================================================
// Qwen3 Model
// ============================================================================

#[derive(Debug)]
struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
}

impl Qwen3Model {
    fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;

        let vb_layers = vb.pp("layers");
        let layers: Vec<_> = (0..config.num_hidden_layers)
            .map(|i| Qwen3Layer::new(config, vb_layers.pp(i)))
            .collect::<Result<_>>()?;

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self { embed_tokens, layers, norm })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;

        for layer in &mut self.layers {
            x = layer.forward(&x, mask, start_pos)?;
        }

        self.norm.forward(&x)
    }

    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

// ============================================================================
// Qwen3 for Causal LM
// ============================================================================

#[derive(Debug)]
pub struct Qwen3ForCausalLM {
    model: Qwen3Model,
    lm_head: Linear,
}

impl Qwen3ForCausalLM {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        // Try loading with "model." prefix first (chat models),
        // then without prefix (embedding models)
        let model = Qwen3Model::new(config, vb.pp("model"))
            .or_else(|_| Qwen3Model::new(config, vb.clone()))?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            // Try both with and without prefix
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))
                .unwrap_or_else(|_| {
                    // For embedding models, create a dummy lm_head using embeddings
                    Linear::new(model.embed_tokens.embeddings().clone(), None)
                })
        };

        Ok(Self { model, lm_head })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;

        let mask = if seq_len > 1 {
            Some(self.causal_mask(seq_len, input_ids.device())?)
        } else {
            None
        };

        let hidden = self.model.forward(input_ids, mask.as_ref(), start_pos)?;
        self.lm_head.forward(&hidden)
    }

    /// Get hidden states without applying lm_head (for embeddings)
    pub fn forward_hidden(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;

        // No causal mask for embeddings - we want bidirectional attention
        // But for simplicity, use causal mask (can be improved)
        let mask = if seq_len > 1 {
            Some(self.causal_mask(seq_len, input_ids.device())?)
        } else {
            None
        };

        self.model.forward(input_ids, mask.as_ref(), 0)
    }

    fn causal_mask(&self, size: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..size)
            .flat_map(|i| (0..size).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(mask, (1, 1, size, size), device)
    }

    pub fn clear_cache(&mut self) {
        self.model.clear_cache();
    }
}

// ============================================================================
// Config Loader
// ============================================================================

pub fn load_config(path: &std::path::Path) -> Result<Qwen3Config> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read config: {}", e)))?;
    serde_json::from_str(&s)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))
}
