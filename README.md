# Qwen3 Inference

A Rust-based super light-weight inference engine for Qwen3 small models (chat and embedding) with CPU optimization.
You could run it in a 2 cores old cpu or cloud server without calling external APIs

## Supported Platforms

| Platform | Architecture | BLAS Backend |
|----------|--------------|--------------|
| macOS | ARM64 (M1/M2/M3) | Accelerate |
| macOS | x86_64 | Accelerate |
| Ubuntu/Linux | x86_64 | MKL or None |
| Ubuntu/Linux | ARM64 | None |

## Build Instructions

### Prerequisites

1. **Install Rust** (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### macOS (Recommended)

```bash
# With Apple Accelerate (optimized BLAS)
cargo build --release --features accelerate
```

### Ubuntu/Linux x86_64

**Option 1: Without BLAS (simplest, works everywhere)**
```bash
cargo build --release
```

**Option 2: With Intel MKL (faster, works on AMD too)**
```bash
# Install MKL first
# For Ubuntu:
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo sh -c 'echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/intel-oneapi.list'
sudo apt update
sudo apt install intel-oneapi-mkl-devel

# Set environment
source /opt/intel/oneapi/setvars.sh

# Build with MKL
cargo build --release --features mkl
```

## Usage

1. **Edit configuration** in `config.toml`:
```toml
[model]
name = "Qwen/Qwen3-0.6B"  # or "Qwen/Qwen3-Embedding-0.6B"
type = "chat"              # or "embedding"

[generation]
max_tokens = 500
temperature = 0.7

[prompt]
user = "Your question here"
no_think = true
```

2. **Run inference**:
```bash
./target/release/qwen3-infer
```

## Supported Models

| Model | Type | Size |
|-------|------|------|
| `Qwen/Qwen3-0.6B` | chat | ~1.4GB |
| `Qwen/Qwen3-1.7B` | chat | ~3.5GB |
| `Qwen/Qwen3-Embedding-0.6B` | embedding | ~1.1GB |

Models are downloaded from ModelScope and cached locally.

## Performance

Expected performance on CPU (tokens/second for generation):

| CPU | No BLAS | With MKL/Accelerate |
|-----|---------|---------------------|
| Apple M1/M2/M3 | ~2-3 | ~4-6 |
| AMD EPYC | ~1-2 | ~3-5 |
| Intel Xeon | ~1-2 | ~3-5 |

## Configuration Options

### Model Settings
```toml
[model]
name = "Qwen/Qwen3-0.6B"    # Model name on ModelScope
type = "chat"                # "chat" or "embedding"
source = "modelscope.cn"     # Download source
dtype = "f32"                # "f32" or "f16"
```

### Generation Settings (chat only)
```toml
[generation]
max_tokens = 500      # Maximum tokens to generate
temperature = 0.7     # 0.0 = deterministic, higher = more random
top_p = 0.9          # Nucleus sampling threshold
seed = 42            # Random seed
```

### Prompt Settings
```toml
[prompt]
system = ""                  # Optional system prompt
user = "Your question"       # User message
no_think = true             # Disable Qwen3 thinking mode
```
