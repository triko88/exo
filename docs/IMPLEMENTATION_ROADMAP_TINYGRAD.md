# Implementation Roadmap: CPU → WebGPU → ROCm (Tinygrad Edition)
## Practical Implementation Guide for Multi-Backend Support with Tinygrad

**Version:** 2.0
**Date:** 2026-01-15
**Priority Order**: CPU (Universal) → WebGPU (Browser) → ROCm (AMD GPUs)
**Framework**: Tinygrad (multi-backend by design)

---

## Executive Summary

This roadmap implements multi-backend support using **tinygrad**, a lightweight ML framework with native multi-backend support. Tinygrad's design philosophy aligns perfectly with exo's cross-platform goals.

**Backend Priority**:
1. **CPU Backend** (Weeks 1-3): Universal support via tinygrad CPU backend
2. **WebGPU Backend** (Weeks 4-6): Browser inference via tinygrad WebGPU backend
3. **ROCm Backend** (Weeks 7-9): AMD GPU support via tinygrad HIP/ROCm backend

**Key Advantage**: Tinygrad provides unified API across all backends - write once, run anywhere.

---

## Why Tinygrad?

### Tinygrad Architecture Benefits

```
┌─────────────────────────────────────────┐
│         Tinygrad Unified API            │
│  (Tensor ops, autograd, model loading)  │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┬──────────┬──────────┐
    │                     │          │          │
┌───▼────┐  ┌────▼─────┐ ┌──▼───┐  ┌──▼───┐  ┌──▼────┐
│  CPU   │  │  Metal   │ │ CUDA │  │ HIP  │  │WebGPU │
│Backend │  │ Backend  │ │Backend│ │Backend│ │Backend│
└────────┘  └──────────┘ └──────┘  └──────┘  └───────┘
```

**Advantages over PyTorch**:
- ✅ **Simpler**: ~5000 LOC vs PyTorch's millions
- ✅ **Multi-backend native**: Backend switching is built-in
- ✅ **Lightweight**: Minimal dependencies
- ✅ **Hackable**: Easy to understand and modify
- ✅ **Unified API**: Same code works on all backends
- ✅ **WebGPU support**: First-class browser support

---

## Phase 1: CPU Backend with Tinygrad (Weeks 1-3)

### Week 1: Foundation + Tinygrad Integration

#### Task 1.1: Install Tinygrad (Day 1)

```toml
# pyproject.toml

[tool.poetry.dependencies]
tinygrad = "^0.9.0"  # Latest stable

# Optional: Specific backends
[tool.poetry.group.gpu]
optional = true
[tool.poetry.group.gpu.dependencies]
# ROCm/HIP support (installed separately on system)
# WebGPU support (comes with tinygrad)
```

```bash
# Install tinygrad
pip install tinygrad

# Verify installation
python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
```

#### Task 1.2: Create Engine Factory (Day 1)

```python
# src/exo/worker/engines/engine_factory.py

from typing import Callable, Any
from exo.shared.types.worker.instances import Instance, MlxRingInstance, MlxJacclInstance

def get_engine_for_instance(instance: Instance):
    """
    Return (initialize_fn, load_fn, generate_fn, warmup_fn) for backend.

    Tinygrad makes this even simpler - same code, different Device!
    """

    backend = _detect_backend(instance)

    if backend == "mlx":
        from exo.worker.engines.mlx.utils_mlx import initialize_mlx, load_mlx_items
        from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
        return initialize_mlx, load_mlx_items, mlx_generate, warmup_inference

    elif backend == "cpu":
        from exo.worker.engines.tinygrad_cpu.utils_tinygrad import initialize_tinygrad, load_tinygrad_items
        from exo.worker.engines.tinygrad_cpu.generator.generate import tinygrad_generate, warmup_inference
        return initialize_tinygrad, load_tinygrad_items, tinygrad_generate, warmup_inference

    elif backend == "webgpu":
        from exo.worker.engines.tinygrad_webgpu.utils_tinygrad import initialize_tinygrad, load_tinygrad_items
        from exo.worker.engines.tinygrad_webgpu.generator.generate import tinygrad_generate, warmup_inference
        return initialize_tinygrad, load_tinygrad_items, tinygrad_generate, warmup_inference

    elif backend == "rocm":
        from exo.worker.engines.tinygrad_rocm.utils_tinygrad import initialize_tinygrad, load_tinygrad_items
        from exo.worker.engines.tinygrad_rocm.generator.generate import tinygrad_generate, warmup_inference
        return initialize_tinygrad, load_tinygrad_items, tinygrad_generate, warmup_inference

    else:
        raise ValueError(f"Unsupported backend: {backend}")

def _detect_backend(instance: Instance) -> str:
    """Auto-detect backend from instance."""
    if hasattr(instance, 'backend_name'):
        return instance.backend_name
    if isinstance(instance, (MlxRingInstance, MlxJacclInstance)):
        return "mlx"
    return "cpu"
```

#### Task 1.3: Update Runner (Day 1)

Same 5-line change as before:

```python
# src/exo/worker/runner/runner.py

from exo.worker.engines.engine_factory import get_engine_for_instance

def main(bound_instance, event_sender, task_receiver):
    initialize_fn, load_fn, generate_fn, warmup_fn = get_engine_for_instance(bound_instance.instance)

    # Rest of code uses these functions (unchanged)
    group = initialize_fn(bound_instance)
    model, tokenizer = load_fn(bound_instance, group)
    # ... etc
```

#### Task 1.4: Add Tinygrad Instance Types (Day 2)

```python
# src/exo/shared/types/worker/instances.py

@dataclass
class TinygradInstance(BaseInstance):
    """Tinygrad inference instance (unified across CPU/GPU/WebGPU)."""
    backend_name: str  # "cpu", "webgpu", "rocm", "cuda"

    # Tinygrad device string (e.g., "CPU", "GPU:0", "WEBGPU", "HIP:0")
    device: str

    # Optional: Distributed setup (for future multi-GPU)
    hosts_by_node: dict[NodeId, list[Host]] | None = None
    ephemeral_port: int | None = None

# Update Instance union
Instance = (
    MlxRingInstance
    | MlxJacclInstance
    | TinygradInstance
)
```

### Week 2: CPU Backend Implementation

#### Task 2.1: Implement Tinygrad CPU Utils (Days 3-5)

```python
# src/exo/worker/engines/tinygrad_cpu/utils_tinygrad.py

import os
from pathlib import Path
from tinygrad import Tensor, Device
from tinygrad.nn.state import load_state_dict, safe_load
from exo.shared.types.worker.instances import BoundInstance, TinygradInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.tinygrad_cpu.model_loader import load_model_from_hf
from exo.worker.runner.bootstrap import logger

def initialize_tinygrad(bound_instance: BoundInstance) -> None:
    """
    Initialize tinygrad backend.

    Tinygrad uses Device.DEFAULT which we set via environment.
    """
    instance = bound_instance.instance

    if not isinstance(instance, TinygradInstance):
        raise ValueError(f"Expected TinygradInstance, got {type(instance)}")

    # Set tinygrad device
    Device.DEFAULT = instance.device
    logger.info(f"Tinygrad initialized with device: {instance.device}")

    # Optimize for CPU if needed
    if instance.device == "CPU":
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)

    return None  # No distributed group for single-device

def load_tinygrad_items(
    bound_instance: BoundInstance,
    group: None
) -> tuple[dict, Any]:
    """
    Load model for tinygrad inference.

    Returns:
        (model_dict, tokenizer): model_dict contains tinygrad Tensors
    """
    shard_meta = bound_instance.bound_shard
    model_id = shard_meta.model_meta.model_id

    # Get model path
    hf_model_path = build_model_path(model_id)

    logger.info(f"Loading model from {hf_model_path}")

    # Load model weights (safetensors or pickle)
    model = load_model_from_hf(hf_model_path, model_id)

    # Load tokenizer (use HuggingFace tokenizers library)
    from tokenizers import Tokenizer
    tokenizer_path = hf_model_path / "tokenizer.json"

    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        # Fallback to transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    logger.info(f"Model loaded successfully: {model_id}")

    return model, tokenizer
```

#### Task 2.2: Model Loader for Tinygrad (Days 3-5)

```python
# src/exo/worker/engines/tinygrad_cpu/model_loader.py

from pathlib import Path
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import Tensor
from exo.shared.types.models import ModelId
from exo.worker.runner.bootstrap import logger

def load_model_from_hf(model_path: Path, model_id: ModelId) -> dict:
    """
    Load HuggingFace model for tinygrad.

    Tinygrad can load safetensors directly!
    """

    # Look for safetensors files
    safetensors_files = list(model_path.glob("*.safetensors"))

    if safetensors_files:
        logger.info(f"Loading from safetensors: {safetensors_files}")

        # Load all shards
        model_dict = {}
        for shard_file in safetensors_files:
            shard = safe_load(str(shard_file))
            model_dict.update(shard)

        logger.info(f"Loaded {len(model_dict)} tensors from safetensors")
        return model_dict

    # Fallback: Load from pytorch bin files
    pytorch_files = list(model_path.glob("*.bin"))
    if pytorch_files:
        logger.info(f"Loading from pytorch bin: {pytorch_files}")

        import torch
        model_dict = {}
        for bin_file in pytorch_files:
            state = torch.load(bin_file, map_location="cpu")
            # Convert torch tensors to tinygrad tensors
            for key, value in state.items():
                model_dict[key] = Tensor(value.numpy())

        return model_dict

    raise FileNotFoundError(f"No model weights found in {model_path}")

def build_llama_model(model_dict: dict, config: dict) -> dict:
    """
    Build Llama model architecture from weights.

    Tinygrad models are just dicts of Tensors + forward functions.
    """
    # Model config
    n_layers = config.get("num_hidden_layers", 32)
    hidden_size = config.get("hidden_size", 4096)
    n_heads = config.get("num_attention_heads", 32)
    vocab_size = config.get("vocab_size", 32000)

    # Extract layers
    model = {
        "embed_tokens": model_dict["model.embed_tokens.weight"],
        "layers": [],
        "norm": model_dict["model.norm.weight"],
        "lm_head": model_dict.get("lm_head.weight", model_dict["model.embed_tokens.weight"])  # Tied embeddings
    }

    # Extract transformer layers
    for i in range(n_layers):
        layer = {
            "self_attn": {
                "q_proj": model_dict[f"model.layers.{i}.self_attn.q_proj.weight"],
                "k_proj": model_dict[f"model.layers.{i}.self_attn.k_proj.weight"],
                "v_proj": model_dict[f"model.layers.{i}.self_attn.v_proj.weight"],
                "o_proj": model_dict[f"model.layers.{i}.self_attn.o_proj.weight"],
            },
            "mlp": {
                "gate_proj": model_dict[f"model.layers.{i}.mlp.gate_proj.weight"],
                "up_proj": model_dict[f"model.layers.{i}.mlp.up_proj.weight"],
                "down_proj": model_dict[f"model.layers.{i}.mlp.down_proj.weight"],
            },
            "input_layernorm": model_dict[f"model.layers.{i}.input_layernorm.weight"],
            "post_attention_layernorm": model_dict[f"model.layers.{i}.post_attention_layernorm.weight"],
        }
        model["layers"].append(layer)

    return model
```

#### Task 2.3: Generator for Tinygrad (Days 6-8)

```python
# src/exo/worker/engines/tinygrad_cpu/generator/generate.py

from typing import Iterator
import time
from tinygrad import Tensor, dtypes
from tinygrad.helpers import all_int
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse, GenerationStats
from exo.worker.engines.tinygrad_cpu.inference import forward_pass, sample_token
from exo.worker.runner.bootstrap import logger

def tinygrad_generate(
    model: dict,
    tokenizer,
    task: ChatCompletionTaskParams,
) -> Iterator[GenerationResponse]:
    """
    Generate tokens using tinygrad CPU backend.

    This is autoregressive generation with tinygrad Tensors.
    """

    # Apply chat template
    prompt = _apply_chat_template(task.messages, tokenizer)

    # Tokenize
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(prompt).ids
    else:
        # HuggingFace tokenizer
        input_ids = tokenizer.encode(prompt)

    # Convert to tinygrad Tensor
    input_tensor = Tensor([input_ids], dtype=dtypes.int32)

    # Generation params
    max_tokens = task.max_tokens or 2048
    temperature = task.temperature or 0.7
    top_p = task.top_p or 0.9

    logger.info(f"Tinygrad generation starting: prompt_len={len(input_ids)}, max_tokens={max_tokens}")

    # KV cache for efficient generation
    cache = None

    # Autoregressive generation loop
    token_count = 0
    start_time = time.time()

    for _ in range(max_tokens):
        # Forward pass
        logits, cache = forward_pass(model, input_tensor, cache)

        # Sample next token
        next_token_id = sample_token(logits, temperature=temperature, top_p=top_p)

        # Decode token
        token_text = _decode_token(tokenizer, next_token_id)

        token_count += 1

        # Check for EOS
        is_eos = next_token_id in [tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2]

        # Yield response
        yield GenerationResponse(
            text=token_text,
            token=token_count,
            finish_reason="stop" if is_eos else None,
            stats=GenerationStats(
                prompt_tokens=len(input_ids),
                completion_tokens=token_count,
                total_tokens=len(input_ids) + token_count,
                throughput_tokens_per_sec=token_count / (time.time() - start_time),
            ),
        )

        if is_eos:
            break

        # Prepare next input (just the new token)
        input_tensor = Tensor([[next_token_id]], dtype=dtypes.int32)

    logger.info(f"Generation complete: {token_count} tokens in {time.time() - start_time:.2f}s")

def warmup_inference(model: dict, tokenizer) -> int:
    """Warmup tinygrad inference."""
    logger.info("Warming up tinygrad inference...")

    # Simple forward pass
    input_ids = Tensor([[1, 2, 3]], dtype=dtypes.int32)
    logits, _ = forward_pass(model, input_ids, cache=None)

    return 3  # 3 tokens processed

def _apply_chat_template(messages, tokenizer) -> str:
    """Apply chat template."""
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Fallback: simple ChatML
    prompt = ""
    for msg in messages:
        role = msg.role
        content = msg.content if isinstance(msg.content, str) else msg.content[0].text
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def _decode_token(tokenizer, token_id: int) -> str:
    """Decode single token."""
    if hasattr(tokenizer, 'id_to_token'):
        return tokenizer.id_to_token(token_id)
    else:
        return tokenizer.decode([token_id])
```

#### Task 2.4: Inference Engine (Days 6-8)

```python
# src/exo/worker/engines/tinygrad_cpu/inference.py

from tinygrad import Tensor
from tinygrad.nn import Linear
import math

def forward_pass(model: dict, input_ids: Tensor, cache: list | None = None):
    """
    Forward pass through transformer.

    Args:
        model: Dict of model weights (Tensors)
        input_ids: Input token IDs [batch, seq_len]
        cache: KV cache from previous passes

    Returns:
        (logits, new_cache)
    """

    # Embedding
    x = model["embed_tokens"][input_ids]

    # Initialize cache if needed
    if cache is None:
        cache = [None] * len(model["layers"])

    new_cache = []

    # Transformer layers
    for i, layer in enumerate(model["layers"]):
        x, layer_cache = transformer_layer(x, layer, cache[i])
        new_cache.append(layer_cache)

    # Final norm
    x = rms_norm(x, model["norm"])

    # LM head
    logits = x @ model["lm_head"].T

    return logits, new_cache

def transformer_layer(x: Tensor, layer: dict, cache: tuple | None):
    """Single transformer layer with attention + MLP."""

    # Pre-norm
    residual = x
    x = rms_norm(x, layer["input_layernorm"])

    # Self-attention
    x, new_cache = attention(x, layer["self_attn"], cache)
    x = x + residual

    # MLP
    residual = x
    x = rms_norm(x, layer["post_attention_layernorm"])
    x = mlp(x, layer["mlp"])
    x = x + residual

    return x, new_cache

def attention(x: Tensor, attn: dict, cache: tuple | None):
    """Multi-head self-attention."""

    batch, seq_len, hidden = x.shape

    # Q, K, V projections
    q = x @ attn["q_proj"].T
    k = x @ attn["k_proj"].T
    v = x @ attn["v_proj"].T

    # Append to cache
    if cache is not None:
        k_cache, v_cache = cache
        k = k_cache.cat(k, dim=1)
        v = v_cache.cat(v, dim=1)

    # Scaled dot-product attention (simplified)
    scale = 1.0 / math.sqrt(hidden)
    scores = (q @ k.transpose(-2, -1)) * scale
    attn_weights = scores.softmax(axis=-1)
    out = attn_weights @ v

    # Output projection
    out = out @ attn["o_proj"].T

    return out, (k, v)

def mlp(x: Tensor, mlp: dict):
    """MLP with SwiGLU activation."""
    gate = x @ mlp["gate_proj"].T
    up = x @ mlp["up_proj"].T

    # SwiGLU: gate * silu(up)
    out = gate * silu(up)
    out = out @ mlp["down_proj"].T

    return out

def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6):
    """RMS normalization."""
    variance = (x * x).mean(axis=-1, keepdim=True)
    x = x * (variance + eps).rsqrt()
    return x * weight

def silu(x: Tensor):
    """SiLU activation (x * sigmoid(x))."""
    return x * x.sigmoid()

def sample_token(logits: Tensor, temperature: float = 1.0, top_p: float = 0.9) -> int:
    """Sample next token from logits."""

    # Get last token logits
    logits = logits[0, -1, :]  # [vocab_size]

    # Temperature scaling
    if temperature > 0:
        logits = logits / temperature

    # Softmax to probabilities
    probs = logits.softmax()

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(axis=0)
        mask = cumsum <= top_p
        # Keep at least one token
        mask[0] = True

        # Filter probs
        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum()

        # Sample from filtered
        token_id = filtered_probs.multinomial(num_samples=1).item()
        return sorted_indices[token_id].item()

    # Sample from full distribution
    return probs.multinomial(num_samples=1).item()
```

### Week 3: CPU Testing & Integration

#### Task 3.1: Update Placement Algorithm (Day 9)

```python
# src/exo/master/placement.py

from exo.shared.types.worker.instances import TinygradInstance

def place_instance(command: PlaceInstance, topology, current_instances):
    # ... existing cycle selection ...

    backend = _detect_backend_for_nodes(selected_cycle)
    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    if backend in ["cpu", "webgpu", "rocm"]:
        # Tinygrad instance
        device = _get_tinygrad_device(backend, selected_cycle)
        target_instances[instance_id] = TinygradInstance(
            instance_id=instance_id,
            shard_assignments=shard_assignments,
            backend_name=backend,
            device=device,
        )
    elif backend == "mlx":
        # Existing MLX logic
        if command.instance_meta == InstanceMeta.MlxJaccl:
            target_instances[instance_id] = MlxJacclInstance(...)
        else:
            target_instances[instance_id] = MlxRingInstance(...)

    return target_instances

def _get_tinygrad_device(backend: str, nodes: list[NodeInfo]) -> str:
    """Map backend to tinygrad device string."""
    if backend == "cpu":
        return "CPU"
    elif backend == "webgpu":
        return "WEBGPU"
    elif backend == "rocm":
        # GPU:0 for first AMD GPU
        return "GPU:0"  # Tinygrad auto-detects HIP
    return "CPU"
```

#### Task 3.2: Testing (Days 10-12)

```python
# tests/worker/engines/test_tinygrad_cpu.py

import pytest
from tinygrad import Tensor

def test_tinygrad_available():
    """Test tinygrad is available."""
    t = Tensor([1, 2, 3])
    assert t.numpy().tolist() == [1, 2, 3]

def test_load_model():
    """Test model loading."""
    from exo.worker.engines.tinygrad_cpu.utils_tinygrad import load_tinygrad_items

    # Mock instance
    bound_instance = create_test_tinygrad_instance(backend_name="cpu", device="CPU")

    model, tokenizer = load_tinygrad_items(bound_instance, None)

    assert model is not None
    assert "embed_tokens" in model
    assert tokenizer is not None

def test_generation():
    """Test token generation."""
    from exo.worker.engines.tinygrad_cpu.generator.generate import tinygrad_generate

    # Load model
    model, tokenizer = ...

    # Create task
    task = ChatCompletionTaskParams(
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10,
    )

    # Generate
    responses = list(tinygrad_generate(model, tokenizer, task))
    assert len(responses) > 0
```

---

## Phase 2: WebGPU Backend with Tinygrad (Weeks 4-6)

### Why Tinygrad + WebGPU is Perfect

Tinygrad has **native WebGPU support**! No ONNX conversion needed.

```python
# It's this simple:
from tinygrad import Device
Device.DEFAULT = "WEBGPU"  # That's it!
```

### Week 4: WebGPU Foundation

#### Task 4.1: Verify WebGPU Support (Day 13)

```python
# Test tinygrad WebGPU
from tinygrad import Tensor, Device

Device.DEFAULT = "WEBGPU"
x = Tensor([1, 2, 3])
print(x.numpy())  # Should work in browser with WebGPU enabled
```

#### Task 4.2: WebGPU Instance (Day 13)

```python
# Already done! TinygradInstance with device="WEBGPU"

instance = TinygradInstance(
    instance_id=InstanceId(),
    shard_assignments=...,
    backend_name="webgpu",
    device="WEBGPU",
)
```

#### Task 4.3: WebGPU Backend Implementation (Days 14-16)

**Key insight**: With tinygrad, CPU and WebGPU backends share almost all code!

```python
# src/exo/worker/engines/tinygrad_webgpu/utils_tinygrad.py

# This is almost identical to tinygrad_cpu!
from tinygrad import Tensor, Device
from exo.worker.engines.tinygrad_cpu.utils_tinygrad import (
    initialize_tinygrad,
    load_tinygrad_items,
)

# Can literally just re-export:
__all__ = ['initialize_tinygrad', 'load_tinygrad_items']
```

```python
# src/exo/worker/engines/tinygrad_webgpu/generator/generate.py

# Also nearly identical!
from exo.worker.engines.tinygrad_cpu.generator.generate import (
    tinygrad_generate,
    warmup_inference,
)

__all__ = ['tinygrad_generate', 'warmup_inference']
```

**That's it!** The same code works because tinygrad abstracts the device.

### Week 5-6: WebGPU Browser Integration (Days 17-24)

#### Task 5.1: Create Web Worker (Days 17-20)

```html
<!-- public/worker.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Exo WebGPU Worker</title>
</head>
<body>
    <h1>Exo Distributed Inference - WebGPU Node</h1>
    <div id="status">Initializing...</div>

    <script type="module">
        // Use Pyodide to run Python in browser
        import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.mjs';

        async function main() {
            // Load Pyodide (Python in browser)
            const pyodide = await loadPyodide();

            // Install tinygrad
            await pyodide.loadPackage('micropip');
            await pyodide.runPythonAsync(`
                import micropip
                await micropip.install('tinygrad')
            `);

            // Set WebGPU device
            await pyodide.runPythonAsync(`
                from tinygrad import Device
                Device.DEFAULT = "WEBGPU"
                print("WebGPU device initialized!")
            `);

            // Load exo worker code
            // ... connect to exo cluster via WebSocket ...

            document.getElementById('status').textContent = 'Connected to Exo cluster!';
        }

        main();
    </script>
</body>
</html>
```

#### Task 5.2: WebSocket Bridge (Days 21-24)

```python
# src/exo/worker/engines/tinygrad_webgpu/websocket_bridge.py

import asyncio
import websockets
import json

class WebGPUBridge:
    """Bridge between exo master and browser-based WebGPU worker."""

    async def connect_worker(self, websocket):
        """Handle WebSocket connection from browser."""

        async for message in websocket:
            data = json.loads(message)

            if data['type'] == 'inference_request':
                # Forward to tinygrad generator
                result = await self.run_inference(data['prompt'])
                await websocket.send(json.dumps({
                    'type': 'inference_response',
                    'result': result
                }))

    async def run_inference(self, prompt: str):
        """Run inference using browser's WebGPU."""
        # This executes in browser via Pyodide
        # ... implementation ...
        pass
```

---

## Phase 3: ROCm Backend with Tinygrad (Weeks 7-9)

### Week 7: ROCm Foundation

#### Task 7.1: Verify Tinygrad ROCm Support (Day 25)

```bash
# Install ROCm on system
# Then verify tinygrad GPU detection

python -c "
from tinygrad import Device
Device.DEFAULT = 'GPU'
print('Detected devices:', Device._devices)
"

# Should show HIP devices on AMD GPUs
```

#### Task 7.2: ROCm Instance (Day 25)

```python
# Use TinygradInstance with GPU device

instance = TinygradInstance(
    instance_id=InstanceId(),
    shard_assignments=...,
    backend_name="rocm",
    device="GPU:0",  # GPU:0, GPU:1, etc.
)
```

### Week 8: ROCm Implementation (Days 26-28)

**Amazing news**: ROCm backend is almost free with tinygrad!

```python
# src/exo/worker/engines/tinygrad_rocm/utils_tinygrad.py

from tinygrad import Device
from exo.worker.engines.tinygrad_cpu.utils_tinygrad import (
    initialize_tinygrad,
    load_tinygrad_items,
)

# Override device initialization
def initialize_tinygrad(bound_instance):
    """Initialize ROCm/HIP backend."""
    instance = bound_instance.instance

    # Set GPU device
    Device.DEFAULT = instance.device  # "GPU:0" for first AMD GPU

    # ROCm-specific optimizations
    import os
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

    logger.info(f"ROCm initialized: {Device.DEFAULT}")
    return None

# Everything else is the same!
__all__ = ['initialize_tinygrad', 'load_tinygrad_items']
```

```python
# src/exo/worker/engines/tinygrad_rocm/generator/generate.py

# Just re-export CPU generator!
from exo.worker.engines.tinygrad_cpu.generator.generate import (
    tinygrad_generate,
    warmup_inference,
)

__all__ = ['tinygrad_generate', 'warmup_inference']
```

#### Task 8.1: Multi-GPU Support (Days 29-30)

For distributed ROCm (multiple GPUs), tinygrad supports multi-device:

```python
# src/exo/worker/engines/tinygrad_rocm/distributed.py

from tinygrad import Tensor, Device

def initialize_distributed_rocm(bound_instance):
    """Initialize multi-GPU ROCm."""

    instance = bound_instance.instance
    rank = get_rank_for_runner(bound_instance.bound_runner_id, instance)

    # Set device for this rank
    device = f"GPU:{rank}"
    Device.DEFAULT = device

    # Tinygrad handles synchronization
    logger.info(f"Multi-GPU ROCm: rank={rank}, device={device}")

    return None  # Tinygrad manages distributed ops
```

### Week 9: ROCm Testing (Days 31-35)

```python
# tests/worker/engines/test_tinygrad_rocm.py

import pytest
from tinygrad import Device, Tensor

@pytest.mark.skipif(not _has_rocm(), reason="ROCm not available")
def test_rocm_device():
    """Test ROCm device initialization."""
    Device.DEFAULT = "GPU:0"
    x = Tensor([1, 2, 3])
    assert x.device.startswith("GPU")

@pytest.mark.skipif(not _has_rocm(), reason="ROCm not available")
def test_rocm_inference():
    """Test inference on AMD GPU."""
    from exo.worker.engines.tinygrad_rocm.generator.generate import tinygrad_generate

    # Load model on GPU
    model, tokenizer = ...

    # Generate
    task = ChatCompletionTaskParams(messages=[...], max_tokens=10)
    responses = list(tinygrad_generate(model, tokenizer, task))

    assert len(responses) > 0

def _has_rocm() -> bool:
    try:
        Device.DEFAULT = "GPU"
        return "HIP" in str(Device._devices)
    except:
        return False
```

---

## Code Reuse Summary

With tinygrad, the code is **dramatically simpler**:

| Backend | Unique Code | Shared Code | Ratio |
|---------|-------------|-------------|-------|
| CPU | ~200 LOC | ~800 LOC | 20% unique |
| WebGPU | ~50 LOC | ~950 LOC | 5% unique |
| ROCm | ~100 LOC | ~900 LOC | 10% unique |

**vs PyTorch approach**: 70-80% unique code per backend!

---

## Directory Structure

```
src/exo/worker/engines/
├── engine_factory.py              # 50 LOC
├── tinygrad_cpu/                  # CPU backend (reference)
│   ├── utils_tinygrad.py          # 150 LOC
│   ├── model_loader.py            # 200 LOC
│   ├── inference.py               # 300 LOC
│   └── generator/
│       └── generate.py            # 150 LOC
├── tinygrad_webgpu/               # WebGPU backend
│   ├── utils_tinygrad.py          # 20 LOC (mostly imports)
│   ├── generator/
│   │   └── generate.py            # 20 LOC (re-exports)
│   └── websocket_bridge.py        # 100 LOC (browser-specific)
└── tinygrad_rocm/                 # ROCm backend
    ├── utils_tinygrad.py          # 50 LOC (device init)
    ├── generator/
    │   └── generate.py            # 20 LOC (re-exports)
    └── distributed.py             # 80 LOC (multi-GPU)
```

**Total**: ~1,140 LOC for all three backends! (vs ~5,000+ with PyTorch)

---

## Dependencies

```toml
# pyproject.toml

[tool.poetry.dependencies]
tinygrad = "^0.9.0"
tokenizers = "^0.15.0"  # Fast tokenizers

# Optional: Browser support
[tool.poetry.group.webgpu]
optional = true
[tool.poetry.group.webgpu.dependencies]
websockets = "^12.0"
aiohttp = "^3.9.0"

# No ROCm-specific deps needed! Tinygrad handles it.
```

---

## Testing Strategy

```bash
# CPU backend
DEVICE=CPU pytest tests/worker/engines/test_tinygrad_cpu.py -v

# WebGPU backend (requires browser)
python -m exo.worker.engines.tinygrad_webgpu.server
# Open http://localhost:8080/worker.html

# ROCm backend (requires AMD GPU)
DEVICE=GPU pytest tests/worker/engines/test_tinygrad_rocm.py -v
```

---

## Performance Expectations

| Backend | Hardware | Throughput | Memory | Notes |
|---------|----------|------------|--------|-------|
| **CPU** | 8-core x86 | 10-15 tok/s | 4GB | TinyLlama-1.1B |
| **WebGPU** | RTX 3060 (browser) | 20-30 tok/s | 6GB | TinyLlama-1.1B |
| **ROCm** | MI250X | 60-80 tok/s | 16GB | Llama-2-7B |
| **ROCm Multi** | 2x MI250X | 120-140 tok/s | 32GB | Llama-2-13B |

---

## Migration from PyTorch Plan (If Needed)

If you later want PyTorch for specific models:

```python
# Hybrid approach: tinygrad for multi-backend, PyTorch for specific models

def get_engine_for_instance(instance):
    backend = instance.backend_name

    if backend in ["cpu", "webgpu", "rocm"] and instance.use_tinygrad:
        # Use tinygrad (default)
        return get_tinygrad_engine(backend)
    elif backend == "rocm" and instance.use_pytorch:
        # Use PyTorch for specific cases
        return get_pytorch_engine(backend)
    # ...
```

---

## Advantages of Tinygrad for Exo

1. **Unified Codebase**: One inference engine, all backends
2. **Lightweight**: 5000 LOC vs PyTorch's millions
3. **Fast Iteration**: Easy to debug and modify
4. **Native Multi-Backend**: No framework lock-in
5. **WebGPU First-Class**: Browser inference just works
6. **Educational**: Clean code, easy to understand

---

## Timeline Summary

| Week | Backend | Status | Key Deliverable |
|------|---------|--------|----------------|
| 1 | Foundation | ✅ Ready | Engine factory, runner update |
| 2 | CPU Core | 🔨 Code | Tinygrad CPU inference working |
| 3 | CPU Polish | 🔨 Test | CPU backend production-ready |
| 4 | WebGPU Core | 🔨 Code | WebGPU device working |
| 5-6 | WebGPU Browser | 🔨 Code | Browser workers connected |
| 7 | ROCm Core | 🔨 Code | Single AMD GPU working |
| 8 | ROCm Multi-GPU | 🔨 Code | Multi-GPU distributed working |
| 9 | ROCm Polish | 🔨 Test | ROCm backend production-ready |

**Total: 9 weeks** (vs 10 weeks with PyTorch, and much simpler code!)

---

## Next Steps

1. **Install tinygrad**: `pip install tinygrad`
2. **Verify basics**: Test CPU inference with tinygrad
3. **Implement engine factory**: Start with Task 1.2
4. **Build CPU backend**: Follow Week 2 tasks

Ready to start? Let's implement the engine factory! 🚀

---

**Document Version History**

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-01-15 | Migrated from PyTorch to tinygrad |
| 1.0 | 2026-01-15 | Initial PyTorch-based roadmap |
