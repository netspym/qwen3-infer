//! Qwen3 MoE (Mixture of Experts) Model Implementation
//!
//! This module implements the Qwen3-30B-A3B MoE architecture for CPU inference.
//! MoE models have multiple "expert" MLPs per layer, with a router that selects
//! which experts to use for each token.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;

// ============================================================================
// Configuration
// ============================================================================

/// Qwen3 MoE model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3MoEConfig {
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
    // MoE specific
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
}

fn default_num_experts() -> usize { 128 }
fn default_num_experts_per_tok() -> usize { 8 }
fn default_norm_topk_prob() -> bool { true }

impl Qwen3MoEConfig {
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    #[inline]
    pub fn expert_intermediate_size(&self) -> usize {
        if self.moe_intermediate_size > 0 {
            self.moe_intermediate_size
        } else {
            self.intermediate_size
        }
    }
}

// ============================================================================
// Rotary Position Embedding (RoPE)
// ============================================================================

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Qwen3MoEConfig, dtype: DType, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim;
        let max_seq_len = config.max_position_embeddings;
        let theta = config.rope_theta;

        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;

        let angles = positions
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        let cos = angles.cos()?.to_dtype(dtype)?;
        let sin = angles.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    #[inline]
    fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let head_dim = x.dim(3)?;
        let half_dim = head_dim / 2;

        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&rotated_x1, &rotated_x2], D::Minus1)
    }
}

// ============================================================================
// Multi-Head Attention with GQA
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
    fn new(config: &Qwen3MoEConfig, vb: VarBuilder) -> Result<Self> {
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

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = self.rotary_emb.apply(&q, start_pos)?;
        let k = self.rotary_emb.apply(&k, start_pos)?;

        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = self.expand_kv(&k)?;
        let v = self.expand_kv(&v)?;

        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;

        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

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
// Expert MLP (Single Expert)
// ============================================================================

#[derive(Debug)]
struct Expert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Expert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
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
// Shared Expert MLP
// ============================================================================

#[derive(Debug)]
struct SharedExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SharedExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
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
// MoE Layer (Sparse Mixture of Experts)
// ============================================================================

#[derive(Debug)]
struct SparseMoE {
    gate: Linear,
    experts: Vec<Expert>,
    shared_expert: Option<SharedExpert>,
    shared_expert_gate: Option<Linear>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoE {
    fn new(config: &Qwen3MoEConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let expert_intermediate_size = config.expert_intermediate_size();

        // Router gate
        let gate = linear_no_bias(hidden_size, config.num_experts, vb.pp("gate"))?;

        // Load experts
        let vb_experts = vb.pp("experts");
        let experts: Vec<_> = (0..config.num_experts)
            .map(|i| Expert::new(hidden_size, expert_intermediate_size, vb_experts.pp(i)))
            .collect::<Result<_>>()?;

        // Shared expert (if configured)
        let (shared_expert, shared_expert_gate) = if config.shared_expert_intermediate_size > 0 {
            let se = SharedExpert::new(
                hidden_size,
                config.shared_expert_intermediate_size,
                vb.pp("shared_expert"),
            )?;
            let se_gate = linear_no_bias(hidden_size, 1, vb.pp("shared_expert_gate"))?;
            (Some(se), Some(se_gate))
        } else {
            (None, None)
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        let num_tokens = batch_size * seq_len;

        // Flatten: [batch * seq, hidden]
        let x_flat = x.reshape((num_tokens, hidden_size))?;

        // Compute router logits: [num_tokens, num_experts]
        let router_logits = self.gate.forward(&x_flat)?;

        // Get top-k experts per token
        let router_probs = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Simple top-k selection (process each token)
        let mut output = Tensor::zeros_like(&x_flat)?;

        // For CPU efficiency, process in a simplified manner
        // Get the top-k indices and weights
        let k = self.num_experts_per_tok;

        for token_idx in 0..num_tokens {
            let token_probs = router_probs.i(token_idx)?;
            let token_input = x_flat.i(token_idx)?.unsqueeze(0)?;

            // Get top-k (simplified: iterate through all and keep top k)
            let probs_vec: Vec<f32> = token_probs.to_vec1()?;
            let mut indexed_probs: Vec<(usize, f32)> =
                probs_vec.iter().cloned().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k: Vec<(usize, f32)> = indexed_probs.into_iter().take(k).collect();

            // Normalize top-k weights if configured
            let weight_sum: f32 = top_k.iter().map(|(_, w)| w).sum();
            let normalizer = if self.norm_topk_prob && weight_sum > 0.0 {
                weight_sum
            } else {
                1.0
            };

            // Compute weighted sum of expert outputs
            let mut token_output = Tensor::zeros_like(&token_input)?;
            for (expert_idx, weight) in top_k {
                let expert_output = self.experts[expert_idx].forward(&token_input)?;
                let normalized_weight = weight / normalizer;
                token_output = (token_output + expert_output * normalized_weight as f64)?;
            }

            // Add to output (simplified assignment)
            let token_output_flat = token_output.squeeze(0)?;
            output = output.slice_assign(&[token_idx..token_idx + 1, 0..hidden_size], &token_output_flat.unsqueeze(0)?)?;
        }

        // Add shared expert contribution if present
        if let (Some(shared_expert), Some(shared_gate)) = (&self.shared_expert, &self.shared_expert_gate) {
            let shared_output = shared_expert.forward(&x_flat)?;
            let shared_weight = candle_nn::ops::sigmoid(&shared_gate.forward(&x_flat)?)?;
            output = (output + shared_output.broadcast_mul(&shared_weight)?)?;
        }

        // Reshape back: [batch, seq, hidden]
        output.reshape((batch_size, seq_len, hidden_size))
    }
}

// ============================================================================
// Standard MLP (for non-MoE layers or fallback)
// ============================================================================

#[derive(Debug)]
struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MLP {
    fn new(config: &Qwen3MoEConfig, vb: VarBuilder) -> Result<Self> {
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
// MLP Wrapper (either MoE or standard)
// ============================================================================

#[derive(Debug)]
enum MLPType {
    Standard(Qwen3MLP),
    MoE(SparseMoE),
}

impl MLPType {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            MLPType::Standard(mlp) => mlp.forward(x),
            MLPType::MoE(moe) => moe.forward(x),
        }
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

#[derive(Debug)]
struct Qwen3MoELayer {
    input_layernorm: RmsNorm,
    self_attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: MLPType,
}

impl Qwen3MoELayer {
    fn new(config: &Qwen3MoEConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Qwen3Attention::new(config, vb.pp("self_attn"))?;
        let post_attention_layernorm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        // Try to load as MoE first, fall back to standard MLP
        let mlp = if config.num_experts > 1 {
            match SparseMoE::new(config, vb.pp("mlp")) {
                Ok(moe) => MLPType::MoE(moe),
                Err(_) => {
                    // Fall back to standard MLP
                    MLPType::Standard(Qwen3MLP::new(config, vb.pp("mlp"))?)
                }
            }
        } else {
            MLPType::Standard(Qwen3MLP::new(config, vb.pp("mlp"))?)
        };

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, start_pos)?;
        let x = (residual + x)?;

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
// Qwen3 MoE Model
// ============================================================================

#[derive(Debug)]
struct Qwen3MoEModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3MoELayer>,
    norm: RmsNorm,
}

impl Qwen3MoEModel {
    fn new(config: &Qwen3MoEConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;

        let vb_layers = vb.pp("layers");
        let layers: Vec<_> = (0..config.num_hidden_layers)
            .map(|i| Qwen3MoELayer::new(config, i, vb_layers.pp(i)))
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
// Qwen3 MoE for Causal LM
// ============================================================================

#[derive(Debug)]
pub struct Qwen3MoEForCausalLM {
    model: Qwen3MoEModel,
    lm_head: Linear,
}

impl Qwen3MoEForCausalLM {
    pub fn new(config: &Qwen3MoEConfig, vb: VarBuilder) -> Result<Self> {
        let model = Qwen3MoEModel::new(config, vb.pp("model"))
            .or_else(|_| Qwen3MoEModel::new(config, vb.clone()))?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))
                .unwrap_or_else(|_| {
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

    pub fn forward_hidden(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;

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

pub fn load_moe_config(path: &std::path::Path) -> Result<Qwen3MoEConfig> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read config: {}", e)))?;
    serde_json::from_str(&s)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))
}
