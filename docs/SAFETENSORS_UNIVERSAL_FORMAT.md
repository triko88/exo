# Safetensors: Universal Format for Heterogeneous Clusters
## Why Safetensors is the Right Choice for Exo

**Version:** 1.0
**Date:** 2026-01-15
**Key Decision**: Use safetensors as the canonical model format across all backends

---

## Executive Summary

**Decision**: Use **safetensors** as the universal model format for exo, not GGUF or backend-specific formats.

**Why**:
- ✅ MLX already uses safetensors
- ✅ Tinygrad natively supports safetensors
- ✅ HuggingFace Hub standard format
- ✅ Same weights work across Mac (MLX) + Linux (tinygrad) nodes
- ✅ Critical for heterogeneous cluster stability

---

## The Heterogeneous Cluster Challenge

### Scenario: Mixed Mac + Linux Cluster

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Mac Node 1  │────│ Linux Node 1 │────│  Mac Node 2  │
│  MLX Backend │     │ Tinygrad CPU │     │  MLX Backend │
│              │     │              │     │              │
│  Llama-7B    │     │  Llama-7B    │     │  Llama-7B    │
│  Layers 0-10 │     │ Layers 11-21 │     │ Layers 22-31 │
└──────────────┘     └──────────────┘     └──────────────┘
```

**Problem**: What if Mac uses `.npz` (MLX format) and Linux uses `.gguf` (llama.cpp)?
- ❌ Different file formats
- ❌ Can't share model cache
- ❌ Tensor shapes might differ
- ❌ Quantization formats incompatible
- ❌ Pipeline parallelism breaks!

**Solution**: Everyone uses safetensors
- ✅ Same file format
- ✅ Shared model cache
- ✅ Guaranteed tensor compatibility
- ✅ Works with MLX + tinygrad quantization
- ✅ Pipeline parallelism works seamlessly

---

## Why Safetensors?

### 1. Universal Support

| Backend | Safetensors Support | Native Format |
|---------|-------------------|---------------|
| **MLX** | ✅ Yes (via mlx-lm) | `.npz`, `.safetensors` |
| **Tinygrad** | ✅ Yes (built-in) | `.safetensors` |
| **PyTorch** | ✅ Yes | `.pt`, `.safetensors` |
| **GGUF** | ⚠️ Conversion needed | `.gguf` |

### 2. HuggingFace Hub Standard

```python
# Most models on HuggingFace Hub already use safetensors
from huggingface_hub import list_repo_files

files = list_repo_files("meta-llama/Llama-2-7b-hf")
# Output: ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors', ...]
```

### 3. Safety & Security

**Safetensors vs Pickle (.bin, .pt)**:
- ✅ No arbitrary code execution (safe to load)
- ✅ Memory-mapped loading (fast)
- ✅ Zero-copy deserialization
- ❌ Pickle can execute malicious code

### 4. Cross-Platform Tensor Format

Safetensors guarantees:
- ✅ Same endianness across platforms
- ✅ Same tensor shapes
- ✅ Same data types
- ✅ Metadata preserved

---

## Tinygrad Safetensors Support

Tinygrad has **native safetensors support**:

```python
# src/exo/worker/engines/tinygrad_cpu/utils_tinygrad.py

from tinygrad.nn.state import safe_load  # Built-in!

def load_tinygrad_items(bound_instance, group):
    """Load model from safetensors (tinygrad native format)."""

    model_path = build_model_path(model_id)

    # Load safetensors directly
    safetensors_files = list(model_path.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    # Load all shards (for large models)
    weights = {}
    for shard_file in sorted(safetensors_files):
        logger.info(f"Loading {shard_file.name}")
        shard_weights = safe_load(str(shard_file))  # Tinygrad's loader
        weights.update(shard_weights)

    logger.info(f"Loaded {len(weights)} tensors from safetensors")

    # Build model from weights
    model = build_model_from_weights(weights, config)

    return model, tokenizer
```

**Key Functions**:
- `safe_load(path)` - Load single safetensors file
- `safe_save(tensors, path)` - Save to safetensors
- Memory-mapped by default (efficient for large models)

---

## MLX Safetensors Support

MLX-LM already uses safetensors:

```python
# MLX loads safetensors natively via mlx-lm
from mlx_lm.utils import load_model

# This works with safetensors automatically
model, tokenizer = load_model("meta-llama/Llama-2-7b-hf")
# Loads from model.safetensors files
```

---

## Quantization with Safetensors

### Quantized Safetensors Format

Both MLX and tinygrad support **quantized safetensors**:

```
model.safetensors structure:
├── model.layers.0.self_attn.q_proj.weight (float16 or int8)
├── model.layers.0.self_attn.q_proj.weight.scale (quantization scale)
├── model.layers.0.self_attn.k_proj.weight
└── ...
```

### MLX Quantization (Current)

```python
# MLX quantizes to safetensors
from mlx_lm.utils import load_model

model, tokenizer = load_model(
    "meta-llama/Llama-2-7b-hf",
    quantize=True  # 4-bit or 8-bit
)
# Internally saves quantized weights to safetensors
```

### Tinygrad Quantization (New)

```python
# Tinygrad can load MLX's quantized safetensors!
from tinygrad.nn.state import safe_load

# Load quantized weights
weights = safe_load("model.safetensors")

# Tinygrad auto-detects quantization from tensor dtype
# int8 tensors → dequantize during forward pass
```

---

## Heterogeneous Cluster Workflow

### Step 1: Download Model (Safetensors)

```python
# src/exo/worker/download/download_utils.py

def build_model_path(model_id: ModelId) -> Path:
    """Download model in safetensors format."""

    from huggingface_hub import snapshot_download

    cache_dir = Path.home() / ".cache" / "exo" / "models"

    # Download only safetensors files (skip pickle)
    model_path = snapshot_download(
        repo_id=str(model_id),
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        ignore_patterns=["*.bin", "*.pt", "*.gguf"],  # Skip other formats
    )

    return Path(model_path)
```

### Step 2: MLX Node Loads Safetensors

```python
# Mac Node (MLX backend)
from mlx_lm.utils import load_model

# Loads safetensors from shared cache
model, tokenizer = load_model("/Users/shared/.cache/exo/models/Llama-2-7b-hf")
```

### Step 3: Tinygrad Node Loads Same Safetensors

```python
# Linux Node (tinygrad backend)
from tinygrad.nn.state import safe_load

# Loads SAME safetensors from shared cache (or NFS mount)
weights = safe_load("/home/shared/.cache/exo/models/Llama-2-7b-hf/model.safetensors")
model = build_model(weights)
```

### Step 4: Pipeline Parallelism Across Backends

```
┌─────────────────────────────────────────────────────────┐
│         Llama-2-7B (32 transformer layers)              │
└─────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
┌────────────────┐   ┌───────────────┐   ┌────────────────┐
│   Mac (MLX)    │   │ Linux (Tinygrad)│  │   Mac (MLX)    │
│  Layers 0-10   │→→→│  Layers 11-21  │→→→│  Layers 22-31  │
│  safetensors   │   │  safetensors   │   │  safetensors   │
└────────────────┘   └────────────────┘   └────────────────┘
```

**Key**: All nodes read from same safetensors files, ensuring:
- ✅ Identical weight values
- ✅ Identical tensor shapes
- ✅ Identical layer boundaries
- ✅ Activations pass correctly between nodes

---

## Model Cache Strategy

### Shared Cache (NFS/Shared Volume)

```
/shared/cache/exo/models/
├── meta-llama--Llama-2-7b-hf/
│   ├── config.json
│   ├── model-00001-of-00002.safetensors  ← All nodes read this
│   ├── model-00002-of-00002.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── mistralai--Mistral-7B-v0.1/
    └── ...
```

**Benefits**:
- ✅ Download once, use everywhere
- ✅ No format conversion needed
- ✅ Consistent across all backend types

### Local Cache with Sync

```bash
# Download on one node
exo download meta-llama/Llama-2-7b-hf

# Sync to other nodes
rsync -av ~/.cache/exo/models/ node2:~/.cache/exo/models/
```

---

## Quantization Compatibility

### Scenario: MLX Quantized, Tinygrad Loads

**MLX side** (Mac):
```python
# Quantize and save to safetensors
from mlx_lm.utils import load_model, save_model

model, tokenizer = load_model("meta-llama/Llama-2-7b-hf", quantize=True)
save_model(model, "~/.cache/exo/models/Llama-2-7b-hf-4bit")
# Saves quantized weights to safetensors
```

**Tinygrad side** (Linux):
```python
# Load MLX's quantized safetensors
from tinygrad.nn.state import safe_load

weights = safe_load("~/.cache/exo/models/Llama-2-7b-hf-4bit/model.safetensors")

# Check for quantization metadata
for key in weights:
    if ".scale" in key:
        print(f"Quantized layer detected: {key}")

# Tinygrad can dequantize during forward pass
model = build_model(weights, quantized=True)
```

**Compatibility**:
- ✅ MLX's 4-bit/8-bit quantization uses standard formats
- ✅ Safetensors preserves quantization metadata
- ✅ Tinygrad can interpret quantized tensors
- ⚠️ May need to implement matching dequantization logic

---

## Testing Heterogeneous Clusters

```python
# tests/integration/test_heterogeneous_safetensors.py

def test_mlx_tinygrad_pipeline():
    """Test pipeline parallelism across MLX and tinygrad with safetensors."""

    # Set up cluster
    cluster = TestCluster([
        Node(platform="darwin", backend="mlx", rank=0),
        Node(platform="linux", backend="cpu", rank=1),
    ])

    # Download model (safetensors)
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    download_model(model_id)

    # Verify both nodes can load
    for node in cluster.nodes:
        weights = node.load_model(model_id)
        assert "model.layers.0.self_attn.q_proj.weight" in weights

    # Run inference with pipeline parallelism
    response = cluster.chat_completion("Hello!")
    assert response.status_code == 200

    # Verify activations passed correctly between MLX and tinygrad
    assert cluster.verify_activations_match()


def test_quantized_safetensors_compatibility():
    """Test MLX quantized model loads in tinygrad."""

    # Quantize with MLX
    mlx_model = quantize_with_mlx("Llama-2-7b-hf", bits=8)
    mlx_model.save_safetensors("model-int8.safetensors")

    # Load with tinygrad
    tinygrad_weights = safe_load("model-int8.safetensors")

    # Verify tensors match
    assert tinygrad_weights["model.embed_tokens.weight"].dtype == dtypes.int8
    assert "model.embed_tokens.weight.scale" in tinygrad_weights
```

---

## Migration Path (If You Have GGUF Models)

### Convert GGUF → Safetensors (One-Time)

```python
# tools/convert_gguf_to_safetensors.py

from tinygrad.nn.state import load_gguf, safe_save
import safetensors

def convert_gguf_to_safetensors(gguf_path: Path, output_path: Path):
    """Convert GGUF model to safetensors for exo compatibility."""

    # Load GGUF
    print(f"Loading GGUF: {gguf_path}")
    weights = load_gguf(str(gguf_path))

    # Convert tinygrad Tensors to numpy
    safetensors_dict = {
        key: tensor.numpy() for key, tensor in weights.items()
    }

    # Save as safetensors
    print(f"Saving safetensors: {output_path}")
    safe_save(safetensors_dict, str(output_path))

    print(f"✓ Converted {len(weights)} tensors")


# Usage
convert_gguf_to_safetensors(
    Path("model-q4_k_m.gguf"),
    Path("model.safetensors")
)
```

---

## Updated Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Exo Distributed Inference Cluster             │
└─────────────────────────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │   Model Download    │
                │  (Safetensors Only) │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐   ┌──────▼───────┐   ┌──────▼───────┐
│  MLX Node    │   │ Tinygrad CPU │   │ Tinygrad GPU │
│ (Mac M3)     │   │ (Linux x86)  │   │ (Linux AMD)  │
│              │   │              │   │              │
│ Loads .safe- │   │ Loads .safe- │   │ Loads .safe- │
│ tensors      │   │ tensors      │   │ tensors      │
└──────────────┘   └──────────────┘   └──────────────┘
```

**Universal Format**: All backends load from same safetensors files

---

## Recommendation

✅ **DO**:
- Use safetensors as the canonical format
- Download only `.safetensors` files from HuggingFace
- Implement quantization within safetensors format
- Test cross-backend compatibility regularly

❌ **DON'T**:
- Use GGUF (unless converting to safetensors)
- Use pickle (.bin, .pt) - security risk
- Use backend-specific formats (.npz for MLX only)
- Mix formats across nodes in same cluster

---

## Code Changes

### Update Download Utils

```python
# src/exo/worker/download/download_utils.py

def build_model_path(model_id: ModelId) -> Path:
    """Download model ensuring safetensors format."""

    model_path = snapshot_download(
        repo_id=str(model_id),
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.npz"],  # Exclude all other formats
    )

    # Validate safetensors exist
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(
            f"Model {model_id} does not have safetensors format. "
            f"Please use a model with safetensors support."
        )

    return Path(model_path)
```

### Update Tinygrad Loader

```python
# src/exo/worker/engines/tinygrad_cpu/utils_tinygrad.py

from tinygrad.nn.state import safe_load

def load_tinygrad_items(bound_instance, group):
    """Load model from safetensors (universal format)."""

    model_path = build_model_path(model_id)

    # Load all safetensors shards
    weights = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        logger.info(f"Loading shard: {shard.name}")
        weights.update(safe_load(str(shard)))

    # Build model (use tinygrad's examples)
    from tinygrad.examples.llama import Transformer
    model = Transformer.from_pretrained(weights)

    return model, tokenizer
```

---

## Performance Comparison

| Format | Load Time | Memory | Cross-Backend |
|--------|-----------|--------|---------------|
| **Safetensors** | Fast (mmap) | Efficient | ✅ Yes |
| GGUF | Fast | Very efficient | ⚠️ Needs conversion |
| Pickle (.bin) | Slow | Inefficient | ❌ Security risk |
| NPZ (MLX) | Medium | Medium | ❌ MLX only |

---

## Summary

**Key Decision**: Use safetensors for all backends
- ✅ Universal support (MLX + tinygrad)
- ✅ HuggingFace standard
- ✅ Safe and efficient
- ✅ Critical for heterogeneous clusters
- ✅ Quantization compatible

**Implementation**:
1. Download only safetensors files
2. Both MLX and tinygrad load from same files
3. Pipeline parallelism works seamlessly
4. Test cross-backend compatibility

🚀 **This ensures stability when testing heterogeneous networks!**

---

**Next Steps**:
1. Update `download_utils.py` to enforce safetensors
2. Verify MLX loads safetensors (already does via mlx-lm)
3. Implement tinygrad safetensors loader
4. Test Mac + Linux mixed cluster
5. Document safetensors as the official format
