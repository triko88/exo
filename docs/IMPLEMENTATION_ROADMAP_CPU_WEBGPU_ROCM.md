# Implementation Roadmap: CPU → WebGPU → ROCm
## Practical Implementation Guide for Multi-Backend Support

**Version:** 1.0
**Date:** 2026-01-15
**Priority Order**: CPU (Universal) → WebGPU (Browser) → ROCm (AMD GPUs)

---

## Executive Summary

This roadmap focuses on implementing backends in order of accessibility and impact:

1. **CPU Backend** (Weeks 1-3): Universal support, works on any Linux/Mac machine
2. **WebGPU Backend** (Weeks 4-6): Browser-based inference, edge deployment
3. **ROCm Backend** (Weeks 7-10): AMD GPU support for data centers

**Strategy**: Use the minimal-change approach with engine factory pattern.

---

## Phase 1: CPU Backend (Weeks 1-3)

### Why CPU First?

- ✅ Works on **every machine** (Linux, macOS, Windows future)
- ✅ No special hardware requirements
- ✅ Great for development and testing
- ✅ Enables raspberry pi, edge devices, laptops
- ✅ Mature ecosystem (llama.cpp is battle-tested)

### Week 1: Foundation + Engine Factory

#### Task 1.1: Create Engine Factory (Day 1)

```python
# src/exo/worker/engines/engine_factory.py

from typing import Callable, Any
from exo.shared.types.worker.instances import Instance, MlxRingInstance, MlxJacclInstance

def get_engine_for_instance(instance: Instance):
    """
    Return (initialize_fn, load_fn, generate_fn, warmup_fn) for backend.

    Returns:
        tuple: (initialize, load_model, generate, warmup) functions
    """

    # Detect backend from instance type or backend_name attribute
    backend = _detect_backend(instance)

    if backend == "mlx":
        from exo.worker.engines.mlx.utils_mlx import initialize_mlx, load_mlx_items
        from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
        return initialize_mlx, load_mlx_items, mlx_generate, warmup_inference

    elif backend == "cpu":
        from exo.worker.engines.cpu.utils_cpu import initialize_cpu, load_cpu_items
        from exo.worker.engines.cpu.generator.generate import cpu_generate, warmup_inference
        return initialize_cpu, load_cpu_items, cpu_generate, warmup_inference

    elif backend == "webgpu":
        from exo.worker.engines.webgpu.utils_webgpu import initialize_webgpu, load_webgpu_items
        from exo.worker.engines.webgpu.generator.generate import webgpu_generate, warmup_inference
        return initialize_webgpu, load_webgpu_items, webgpu_generate, warmup_inference

    elif backend == "rocm":
        from exo.worker.engines.rocm.utils_rocm import initialize_rocm, load_rocm_items
        from exo.worker.engines.rocm.generator.generate import rocm_generate, warmup_inference
        return initialize_rocm, load_rocm_items, rocm_generate, warmup_inference

    else:
        raise ValueError(f"Unsupported backend: {backend} for instance type: {type(instance)}")

def _detect_backend(instance: Instance) -> str:
    """Auto-detect backend from instance."""

    # Explicit backend field (preferred)
    if hasattr(instance, 'backend_name'):
        return instance.backend_name

    # Legacy: Infer from instance type
    if isinstance(instance, (MlxRingInstance, MlxJacclInstance)):
        return "mlx"

    # Default fallback
    return "cpu"
```

#### Task 1.2: Update Runner (Day 1)

```python
# src/exo/worker/runner/runner.py (ONLY CHANGE THESE LINES)

# REPLACE lines 42-47:
# OLD:
# from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
# from exo.worker.engines.mlx.utils_mlx import initialize_mlx, load_mlx_items, mlx_force_oom

# NEW:
from exo.worker.engines.engine_factory import get_engine_for_instance

def main(bound_instance, event_sender, task_receiver):
    instance = bound_instance.instance

    # Get engine-specific functions
    initialize_fn, load_fn, generate_fn, warmup_fn = get_engine_for_instance(instance)

    model = None
    tokenizer = None
    group = None

    current_status = RunnerIdle()
    # ... rest stays EXACTLY the same, just replace function calls:

    for task in tasks:
        match task:
            case ConnectToGroup() if isinstance(current_status, (RunnerIdle, RunnerFailed)):
                current_status = RunnerConnecting()
                event_sender.send(RunnerStatusUpdated(...))

                group = initialize_fn(bound_instance)  # Was: initialize_mlx()

                current_status = RunnerConnected()

            case LoadModel() if ...:
                current_status = RunnerLoading()
                event_sender.send(RunnerStatusUpdated(...))

                model, tokenizer = load_fn(bound_instance, group)  # Was: load_mlx_items()

                current_status = RunnerLoaded()

            case StartWarmup() if isinstance(current_status, RunnerLoaded):
                current_status = RunnerWarmingUp()
                event_sender.send(RunnerStatusUpdated(...))

                toks = warmup_fn(model=model, tokenizer=tokenizer)  # Was: warmup_inference()

                current_status = RunnerReady()

            case ChatCompletion(task_params=task_params, command_id=command_id) if ...:
                current_status = RunnerRunning()
                event_sender.send(RunnerStatusUpdated(...))

                for response in generate_fn(model=model, tokenizer=tokenizer, task=task_params):  # Was: mlx_generate()
                    if shard_metadata.device_rank == 0:
                        event_sender.send(ChunkGenerated(...))

                current_status = RunnerReady()

            # ... rest unchanged
```

**Testing**: Verify MLX still works (should be 100% unchanged behavior)

#### Task 1.3: Add CPU Instance Type (Day 2)

```python
# src/exo/shared/types/worker/instances.py

from enum import Enum

class BackendType(str, Enum):
    """Supported inference backends."""
    MLX = "mlx"
    CPU = "cpu"
    WEBGPU = "webgpu"
    ROCM = "rocm"
    CUDA = "cuda"  # Future

@dataclass
class CpuInstance(BaseInstance):
    """CPU-only inference instance (single node or distributed via Gloo)."""
    backend_name: str = "cpu"

    # For single-node CPU
    num_threads: int | None = None  # Auto-detect from CPU count

    # For distributed CPU (optional, future)
    hosts_by_node: dict[NodeId, list[Host]] | None = None
    ephemeral_port: int | None = None

# Update Instance union
Instance = (
    MlxRingInstance
    | MlxJacclInstance
    | CpuInstance
    # | WebGpuInstance  # Add in Phase 2
    # | RocmInstance    # Add in Phase 3
)
```

### Week 2: CPU Backend Implementation

#### Task 2.1: Set Up llama.cpp Integration (Days 3-4)

**Dependencies**:
```toml
# pyproject.toml additions

[tool.poetry.dependencies]
llama-cpp-python = "^0.2.77"  # Python bindings for llama.cpp
```

**Directory Structure**:
```
src/exo/worker/engines/cpu/
├── __init__.py
├── utils_cpu.py           # initialize_cpu, load_cpu_items
├── generator/
│   ├── __init__.py
│   └── generate.py        # cpu_generate, warmup_inference
├── model_converter.py     # Convert HF → GGUF
└── constants.py           # CPU-specific config
```

#### Task 2.2: Implement utils_cpu.py (Days 3-4)

```python
# src/exo/worker/engines/cpu/utils_cpu.py

import os
from pathlib import Path
from llama_cpp import Llama
from exo.shared.types.worker.instances import BoundInstance, CpuInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.cpu.model_converter import ensure_gguf_model
from exo.worker.runner.bootstrap import logger

def initialize_cpu(bound_instance: BoundInstance) -> None:
    """
    Initialize CPU backend.

    For CPU, we don't need distributed setup initially (single-node only).
    Returns None since llama.cpp handles everything internally.
    """
    instance = bound_instance.instance

    if not isinstance(instance, CpuInstance):
        raise ValueError(f"Expected CpuInstance, got {type(instance)}")

    # Set CPU threads
    if instance.num_threads:
        os.environ["OMP_NUM_THREADS"] = str(instance.num_threads)
    else:
        # Auto-detect optimal thread count
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)

    logger.info(f"CPU backend initialized with {os.environ['OMP_NUM_THREADS']} threads")

    return None  # No distributed group for now

def load_cpu_items(
    bound_instance: BoundInstance,
    group: None  # CPU doesn't use distributed group (yet)
) -> tuple[Llama, Llama]:
    """
    Load model with llama.cpp.

    Returns:
        (model, tokenizer): For llama.cpp, both are the same Llama object
    """
    shard_meta = bound_instance.bound_shard
    model_id = shard_meta.model_meta.model_id

    # Get HuggingFace model path
    hf_model_path = build_model_path(model_id)

    # Convert to GGUF if needed (cached)
    gguf_path = ensure_gguf_model(hf_model_path, model_id)

    logger.info(f"Loading GGUF model from {gguf_path}")

    # Load with llama.cpp
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=4096,  # Context window
        n_threads=int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4)),
        n_batch=512,  # Batch size for prompt processing
        use_mlock=True,  # Lock model in RAM (prevent swapping)
        verbose=False,
    )

    logger.info(f"CPU model loaded successfully: {model_id}")

    # llama.cpp handles tokenization internally
    tokenizer = model

    return model, tokenizer
```

#### Task 2.3: Model Converter (Day 4)

```python
# src/exo/worker/engines/cpu/model_converter.py

import subprocess
from pathlib import Path
from exo.shared.types.models import ModelId
from exo.worker.runner.bootstrap import logger

def ensure_gguf_model(hf_model_path: Path, model_id: ModelId) -> Path:
    """
    Convert HuggingFace model to GGUF format (cached).

    Args:
        hf_model_path: Path to HuggingFace model
        model_id: Model identifier

    Returns:
        Path to GGUF file
    """
    # Cache directory
    cache_dir = Path.home() / ".cache" / "exo" / "gguf" / str(model_id).replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)

    gguf_file = cache_dir / "model-q4_k_m.gguf"  # Q4_K_M is good balance

    # Check if already converted
    if gguf_file.exists():
        logger.info(f"Using cached GGUF model: {gguf_file}")
        return gguf_file

    logger.info(f"Converting {model_id} to GGUF format...")

    try:
        # Option 1: Use llama.cpp convert script
        # Requires llama.cpp repo cloned locally
        # subprocess.run([
        #     "python", "llama.cpp/convert.py",
        #     str(hf_model_path),
        #     "--outfile", str(gguf_file),
        #     "--outtype", "q4_k_m"
        # ], check=True)

        # Option 2: Use HuggingFace Hub for pre-converted GGUF models
        # Many models have GGUF versions available
        gguf_model_id = _find_gguf_variant(model_id)
        if gguf_model_id:
            from huggingface_hub import hf_hub_download
            logger.info(f"Downloading pre-converted GGUF from {gguf_model_id}")
            downloaded = hf_hub_download(
                repo_id=str(gguf_model_id),
                filename="*q4_k_m.gguf",  # Adjust pattern as needed
                local_dir=cache_dir,
            )
            return Path(downloaded)

        # Option 3: Fallback - download from HF Hub GGUF repos
        raise NotImplementedError(
            f"GGUF conversion not yet implemented. "
            f"Please manually convert {model_id} to GGUF or use a pre-converted variant."
        )

    except Exception as e:
        logger.error(f"GGUF conversion failed: {e}")
        raise

def _find_gguf_variant(model_id: ModelId) -> ModelId | None:
    """
    Find GGUF variant of a model on HuggingFace Hub.

    Common patterns:
    - meta-llama/Llama-2-7b → TheBloke/Llama-2-7B-GGUF
    - mistralai/Mistral-7B-v0.1 → TheBloke/Mistral-7B-v0.1-GGUF
    """
    # Simple heuristic: TheBloke usually has GGUF versions
    model_name = str(model_id).split("/")[-1]
    candidate = f"TheBloke/{model_name}-GGUF"

    # TODO: Actually check if it exists on HF Hub
    logger.info(f"Trying GGUF variant: {candidate}")

    return ModelId(candidate)
```

#### Task 2.4: Generator (Days 5-6)

```python
# src/exo/worker/engines/cpu/generator/generate.py

from typing import Iterator
from llama_cpp import Llama
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse, GenerationStats
from exo.worker.runner.bootstrap import logger

def cpu_generate(
    model: Llama,
    tokenizer: Llama,  # Same as model for llama.cpp
    task: ChatCompletionTaskParams,
) -> Iterator[GenerationResponse]:
    """
    Generate tokens using llama.cpp CPU backend.

    Args:
        model: Llama.cpp model instance
        tokenizer: Same as model (llama.cpp handles tokenization)
        task: Chat completion parameters

    Yields:
        GenerationResponse objects with streaming tokens
    """

    # Convert messages to prompt (apply chat template)
    prompt = _apply_chat_template(task.messages, model)

    # Generation parameters
    max_tokens = task.max_tokens or 2048
    temperature = task.temperature or 0.7
    top_p = task.top_p or 0.9

    logger.info(f"CPU generation starting: max_tokens={max_tokens}, temp={temperature}")

    # Stream generation with llama.cpp
    token_count = 0
    full_text = ""

    try:
        for output in model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,  # Enable streaming
            stop=["</s>", "<|im_end|>", "<|endoftext|>"],  # Stop tokens
        ):
            # Extract token text
            token_text = output["choices"][0]["text"]
            full_text += token_text
            token_count += 1

            # Check for finish
            finish_reason = output["choices"][0].get("finish_reason")

            # Yield response
            yield GenerationResponse(
                text=token_text,
                token=token_count,
                finish_reason=finish_reason,
                stats=GenerationStats(
                    prompt_tokens=0,  # llama.cpp doesn't expose this during streaming
                    completion_tokens=token_count,
                    total_tokens=token_count,
                    throughput_tokens_per_sec=0.0,  # Calculate at end
                ),
            )

            # Break if done
            if finish_reason:
                break

    except Exception as e:
        logger.error(f"CPU generation error: {e}")
        raise

    logger.info(f"CPU generation complete: {token_count} tokens generated")

def warmup_inference(model: Llama, tokenizer: Llama) -> int:
    """
    Warmup CPU inference by generating a few tokens.

    Returns:
        Number of tokens generated during warmup
    """
    logger.info("Warming up CPU inference...")

    warmup_prompt = "Hello, world!"

    output = model(
        warmup_prompt,
        max_tokens=10,
        temperature=0.0,
        stream=False,
    )

    tokens_generated = len(output["choices"][0]["text"].split())
    logger.info(f"Warmup complete: {tokens_generated} tokens")

    return tokens_generated

def _apply_chat_template(messages, model: Llama) -> str:
    """
    Apply chat template to messages.

    For now, use simple format. In production, detect model type
    and apply appropriate template (ChatML, Llama2, Mistral, etc.)
    """
    # Simple ChatML format
    prompt = ""
    for msg in messages:
        role = msg.role
        content = msg.content

        if isinstance(content, list):
            # Handle multi-part content (text + images)
            content = " ".join([c.text for c in content if hasattr(c, "text")])
        elif hasattr(content, "text"):
            content = content.text

        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"

    return prompt
```

### Week 3: CPU Testing & Integration

#### Task 3.1: Update Placement Algorithm (Day 7)

```python
# src/exo/master/placement.py

from exo.shared.types.worker.instances import CpuInstance

def place_instance(command: PlaceInstance, topology, current_instances):
    # ... existing cycle selection ...

    # Detect backend for selected cycle
    backend = _detect_backend_for_nodes(selected_cycle)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    if backend == "cpu":
        # CPU instance (single node for now)
        target_instances[instance_id] = CpuInstance(
            instance_id=instance_id,
            shard_assignments=shard_assignments,
            num_threads=None,  # Auto-detect
        )

    elif backend == "mlx":
        # Existing MLX logic unchanged
        if command.instance_meta == InstanceMeta.MlxJaccl:
            target_instances[instance_id] = MlxJacclInstance(...)
        else:
            target_instances[instance_id] = MlxRingInstance(...)

    return target_instances

def _detect_backend_for_nodes(nodes: list[NodeInfo]) -> str:
    """Auto-detect best backend for nodes."""
    # For now, simple heuristic
    first_node = nodes[0]

    if first_node.node_profile and first_node.node_profile.devices:
        device = first_node.node_profile.devices[0]
        return device.backend_name

    # Default to CPU
    return "cpu"
```

#### Task 3.2: Device Detection (Day 8)

```python
# src/exo/worker/utils/device_detection.py

from exo.shared.types.worker.instances import BackendType
from exo.worker.engines.base_device import DeviceCapability, DeviceType
from exo.shared.types.memory import Memory
import psutil

def detect_cpu_device() -> DeviceCapability:
    """Detect CPU capabilities."""
    import platform

    # Get memory info
    mem = psutil.virtual_memory()

    return DeviceCapability(
        device_type=DeviceType.CPU,
        device_index=0,
        device_name=f"{platform.processor()} ({os.cpu_count()} cores)",
        compute_capability=None,
        memory_total=Memory.from_bytes(mem.total),
        memory_available=Memory.from_bytes(mem.available),
        supports_fp16=False,  # CPU uses float32 typically
        supports_bf16=False,
        supports_int8=True,   # GGUF supports int8
        supports_int4=True,   # GGUF supports int4
        backend_name="cpu",
        rdma_capable=False,
        rdma_device_name=None,
    )
```

#### Task 3.3: Testing (Days 9-10)

**Unit Tests**:
```python
# tests/worker/engines/test_cpu_backend.py

def test_cpu_backend_available():
    from exo.worker.engines.cpu.utils_cpu import initialize_cpu
    # CPU should always be available
    assert True

def test_cpu_model_loading(sample_model_path):
    from exo.worker.engines.cpu.utils_cpu import load_cpu_items

    bound_instance = create_test_cpu_instance()
    model, tokenizer = load_cpu_items(bound_instance, None)

    assert model is not None
    assert tokenizer is not None

def test_cpu_generation(sample_model_path):
    from exo.worker.engines.cpu.generator.generate import cpu_generate
    from exo.shared.types.tasks import ChatCompletionTaskParams

    # Load model
    model, tokenizer = load_cpu_items(...)

    # Create task
    task = ChatCompletionTaskParams(
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10,
    )

    # Generate
    responses = list(cpu_generate(model, tokenizer, task))
    assert len(responses) > 0
    assert responses[-1].finish_reason is not None
```

**Integration Test**:
```bash
# Manual test on Linux VM
python -m exo.main --backend cpu --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

---

## Phase 2: WebGPU Backend (Weeks 4-6)

### Why WebGPU Second?

- ✅ Enables **browser-based nodes** (massive potential user base)
- ✅ Works on Chromium-based browsers (Chrome, Edge, Opera)
- ✅ No installation required for end users
- ✅ Great for demos and lightweight inference
- ✅ Leverages GPU on consumer devices (gaming laptops, etc.)

### Week 4: WebGPU Foundation

#### Task 4.1: Add WebGPU Instance Type (Day 11)

```python
# src/exo/shared/types/worker/instances.py

@dataclass
class WebGpuInstance(BaseInstance):
    """WebGPU inference instance (browser-based)."""
    backend_name: str = "webgpu"

    # WebGPU specific config
    worker_url: str | None = None  # URL of web worker
    max_buffer_size: int = 2048  # GPU buffer size limit
    use_wasm: bool = True  # Use WebAssembly for heavy compute

# Update Instance union
Instance = (
    MlxRingInstance
    | MlxJacclInstance
    | CpuInstance
    | WebGpuInstance
)
```

#### Task 4.2: Choose WebGPU Framework (Day 11)

**Options**:
1. **ONNX Runtime Web** (Recommended)
   - Mature, Microsoft-backed
   - Supports WebGPU backend
   - Works with ONNX models

2. **Transformers.js** (Hugging Face)
   - Native HuggingFace integration
   - WebGPU support in development
   - Easier model loading

3. **WebLLM** (MLC-AI)
   - Specialized for LLMs
   - TVM-based compilation
   - Best performance

**Decision**: Start with **ONNX Runtime Web** for stability.

```toml
# pyproject.toml (web worker dependencies separate)

[tool.poetry.group.webgpu]
optional = true

[tool.poetry.group.webgpu.dependencies]
onnxruntime-web = "^1.17.0"
transformers = "^4.37.0"  # For model export
optimum = "^1.16.0"  # For ONNX conversion
```

#### Task 4.3: Implement WebGPU Backend (Days 12-15)

```python
# src/exo/worker/engines/webgpu/utils_webgpu.py

import asyncio
from pathlib import Path
from exo.shared.types.worker.instances import BoundInstance, WebGpuInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.webgpu.model_converter import ensure_onnx_model
from exo.worker.runner.bootstrap import logger

# Note: WebGPU requires async/await, we'll wrap it for sync interface

def initialize_webgpu(bound_instance: BoundInstance) -> None:
    """
    Initialize WebGPU backend.

    WebGPU runs in browser context, so this is mostly validation.
    """
    instance = bound_instance.instance

    if not isinstance(instance, WebGpuInstance):
        raise ValueError(f"Expected WebGpuInstance, got {type(instance)}")

    # Check if we're in a browser-compatible environment
    # For server-side, we'll use headless browser or skip
    logger.info("WebGPU backend initialized (browser context required)")

    return None

def load_webgpu_items(
    bound_instance: BoundInstance,
    group: None
) -> tuple[Any, Any]:
    """
    Load model for WebGPU inference.

    This actually prepares the ONNX model and returns a placeholder.
    Real loading happens in JavaScript context.
    """
    shard_meta = bound_instance.bound_shard
    model_id = shard_meta.model_meta.model_id

    # Get HF model path
    hf_model_path = build_model_path(model_id)

    # Convert to ONNX
    onnx_path = ensure_onnx_model(hf_model_path, model_id)

    logger.info(f"WebGPU model prepared: {onnx_path}")

    # Return paths for now (actual loading in JS)
    model = {"type": "webgpu_onnx", "path": str(onnx_path)}
    tokenizer = {"type": "webgpu_tokenizer", "path": str(hf_model_path)}

    return model, tokenizer
```

```python
# src/exo/worker/engines/webgpu/generator/generate.py

from typing import Iterator
import json
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse, GenerationStats
from exo.worker.runner.bootstrap import logger

def webgpu_generate(
    model: dict,  # Model metadata
    tokenizer: dict,  # Tokenizer metadata
    task: ChatCompletionTaskParams,
) -> Iterator[GenerationResponse]:
    """
    Generate tokens using WebGPU backend.

    This is a bridge - actual inference happens in JavaScript/WASM.
    For now, this is a placeholder for the architecture.
    """

    logger.info("WebGPU generation starting (JS bridge required)")

    # In production, this would:
    # 1. Send request to web worker via WebSocket/MessageChannel
    # 2. Receive streaming tokens back
    # 3. Yield GenerationResponse objects

    # Placeholder implementation
    raise NotImplementedError(
        "WebGPU generation requires JavaScript runtime. "
        "Use the web worker interface for browser-based inference."
    )

def warmup_inference(model: dict, tokenizer: dict) -> int:
    """Warmup WebGPU inference."""
    logger.info("WebGPU warmup (placeholder)")
    return 0
```

### Week 5-6: WebGPU Web Worker (Days 16-21)

This requires implementing a JavaScript/TypeScript web worker:

```typescript
// src/exo/worker/engines/webgpu/worker/webgpu_worker.ts

import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime for WebGPU
ort.env.wasm.wasmPaths = '/path/to/wasm/files/';

class WebGPUInferenceWorker {
    private session: ort.InferenceSession | null = null;
    private tokenizer: any = null;

    async loadModel(modelPath: string, tokenizerPath: string) {
        // Load ONNX model with WebGPU backend
        this.session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['webgpu'],
        });

        // Load tokenizer (use transformers.js or custom)
        this.tokenizer = await loadTokenizer(tokenizerPath);

        console.log('WebGPU model loaded successfully');
    }

    async* generate(prompt: string, maxTokens: number = 100): AsyncGenerator<string> {
        if (!this.session || !this.tokenizer) {
            throw new Error('Model not loaded');
        }

        // Tokenize input
        const inputIds = await this.tokenizer.encode(prompt);

        // Autoregressive generation
        let generatedTokens = 0;
        let currentIds = inputIds;

        while (generatedTokens < maxTokens) {
            // Create input tensor
            const inputTensor = new ort.Tensor('int64', currentIds, [1, currentIds.length]);

            // Run inference
            const outputs = await this.session.run({ input_ids: inputTensor });
            const logits = outputs.logits;

            // Sample next token (greedy for now)
            const nextTokenId = argmax(logits.data);

            // Decode token
            const tokenText = await this.tokenizer.decode([nextTokenId]);
            yield tokenText;

            // Append to context
            currentIds.push(nextTokenId);
            generatedTokens++;

            // Check for EOS
            if (nextTokenId === this.tokenizer.eosTokenId) {
                break;
            }
        }
    }
}

// Message handler for communication with Python
self.onmessage = async (event) => {
    const { type, payload } = event.data;

    switch (type) {
        case 'load':
            await worker.loadModel(payload.modelPath, payload.tokenizerPath);
            self.postMessage({ type: 'loaded' });
            break;

        case 'generate':
            for await (const token of worker.generate(payload.prompt, payload.maxTokens)) {
                self.postMessage({ type: 'token', token });
            }
            self.postMessage({ type: 'done' });
            break;
    }
};

const worker = new WebGPUInferenceWorker();
```

**Note**: WebGPU implementation requires significant JavaScript work. Consider this **Phase 2B** (optional for MVP).

---

## Phase 3: ROCm Backend (Weeks 7-10)

### Why ROCm Third?

- ✅ AMD GPUs gaining market share (MI200, MI300, RDNA3)
- ✅ Similar to CUDA implementation (can reuse patterns)
- ✅ Strong data center presence (AWS, Azure ROCm instances)
- ✅ PyTorch has good ROCm support

### Week 7: ROCm Foundation

#### Task 7.1: Add ROCm Instance Type (Day 22)

```python
# src/exo/shared/types/worker/instances.py

@dataclass
class RocmDistributedInstance(BaseInstance):
    """ROCm backend with RCCL distributed."""
    backend_name: str = "rocm"

    # TCP-based setup (like NCCL)
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int

    # Optional: RDMA config for InfiniBand
    use_rdma: bool = False
    rdma_devices: list[list[str | None]] | None = None

# Update Instance union
Instance = (
    MlxRingInstance
    | MlxJacclInstance
    | CpuInstance
    | WebGpuInstance
    | RocmDistributedInstance
)
```

#### Task 7.2: Dependencies (Day 22)

```toml
# pyproject.toml

[tool.poetry.group.rocm]
optional = true

[tool.poetry.group.rocm.dependencies]
# PyTorch with ROCm (special index)
torch = {version = "^2.1.0", source = "pytorch-rocm"}
transformers = "^4.37.0"
accelerate = "^0.26.0"

[[tool.poetry.source]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm5.7"
priority = "supplemental"
```

### Week 8-9: ROCm Implementation (Days 23-28)

```python
# src/exo/worker/engines/rocm/utils_rocm.py

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from exo.shared.types.worker.instances import BoundInstance, RocmDistributedInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.runner.bootstrap import logger

def initialize_rocm(bound_instance: BoundInstance) -> torch.distributed.ProcessGroup | None:
    """
    Initialize ROCm backend with RCCL distributed.

    Very similar to CUDA/NCCL setup.
    """
    instance = bound_instance.instance

    if not isinstance(instance, RocmDistributedInstance):
        return None  # Single GPU

    # Verify ROCm available
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm not available (torch.cuda.is_available() == False)")

    if "rocm" not in torch.version.hip:
        raise RuntimeError(f"PyTorch not built with ROCm support: {torch.version.hip}")

    # Get rank and world size
    rank = _get_rank_for_runner(bound_instance.bound_runner_id, instance)
    world_size = len(instance.shard_assignments.runner_to_shard)

    # Set up RCCL environment
    master_node = list(instance.hosts_by_node.values())[0][0]
    os.environ["MASTER_ADDR"] = str(master_node.ipv4_address)
    os.environ["MASTER_PORT"] = str(instance.ephemeral_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # ROCm-specific optimizations
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

    if instance.use_rdma:
        # Enable RCCL over InfiniBand
        os.environ["RCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_SOCKET_IFNAME"] = "ib0"  # Adjust for your IB interface

    # Initialize process group with NCCL backend (RCCL is a fork of NCCL)
    dist.init_process_group(
        backend="nccl",  # RCCL uses same API as NCCL
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )

    logger.info(f"ROCm initialized: rank={rank}, world_size={world_size}")

    return dist.group.WORLD

def load_rocm_items(
    bound_instance: BoundInstance,
    group: torch.distributed.ProcessGroup | None
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load model for ROCm inference.

    Almost identical to CUDA implementation.
    """
    shard_meta = bound_instance.bound_shard
    model_id = shard_meta.model_meta.model_id
    model_path = build_model_path(model_id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if group is None:
        # Single GPU
        logger.info(f"Loading model on single ROCm GPU: {torch.cuda.get_device_name(0)}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # ROCm supports FP16 well
            device_map="auto"
        )
    else:
        # Distributed: Apply sharding
        logger.info(f"Loading sharded model for ROCm distributed")
        model = _load_sharded_rocm_model(model_path, shard_meta, group)

    return model, tokenizer

def _load_sharded_rocm_model(model_path, shard_meta, group):
    """
    Apply tensor or pipeline parallelism for ROCm.

    Same logic as CUDA - can reuse!
    """
    from exo.shared.types.worker.shards import TensorShardMetadata, PipelineShardMetadata

    if isinstance(shard_meta, TensorShardMetadata):
        # Tensor parallelism with Accelerate
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_path)

        # Shard linear layers across GPUs
        # (Implementation details similar to CUDA)

    elif isinstance(shard_meta, PipelineShardMetadata):
        # Pipeline parallelism - load specific layers
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )

        # Keep only layers [start_layer:end_layer]
        # (Implementation details similar to CUDA)

    return model
```

```python
# src/exo/worker/engines/rocm/generator/generate.py

# Almost identical to CUDA implementation!
# Can copy-paste with minimal changes

from transformers import TextIteratorStreamer
import threading
import torch

def rocm_generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    task: ChatCompletionTaskParams,
) -> Iterator[GenerationResponse]:
    """Generate with ROCm backend."""

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        task.messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Set up streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # Generate in background
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "max_new_tokens": task.max_tokens or 2048,
        "temperature": task.temperature or 0.7,
        "top_p": task.top_p or 0.9,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream tokens
    token_count = 0
    for text in streamer:
        token_count += 1
        yield GenerationResponse(
            text=text,
            token=token_count,
            finish_reason=None,
            stats=GenerationStats(...)
        )

    thread.join()
    yield GenerationResponse(text="", token=token_count, finish_reason="stop", stats=...)

def warmup_inference(model, tokenizer) -> int:
    """Warmup ROCm inference."""
    warmup_prompt = "Hello"
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    return outputs.shape[1] - inputs.input_ids.shape[1]
```

### Week 10: ROCm Testing (Days 29-30)

**Hardware Requirements**:
- AMD GPU with ROCm support (MI200, MI250, RDNA3)
- ROCm 5.7+ installed
- PyTorch ROCm build

**Testing**:
```bash
# Verify ROCm
rocm-smi

# Test single GPU
python -m exo.main --backend rocm --model "meta-llama/Llama-2-7b-chat"

# Test multi-GPU (2x MI250)
python -m exo.main --backend rocm --distributed --gpus 2
```

---

## Summary Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | CPU Foundation | Engine factory, runner updated, CPU instance type |
| 2 | CPU Implementation | llama.cpp integration, GGUF converter |
| 3 | CPU Testing | Unit tests, integration tests, placement updates |
| 4 | WebGPU Foundation | WebGPU instance type, ONNX conversion |
| 5-6 | WebGPU Implementation | Web worker, JavaScript bridge (optional MVP) |
| 7 | ROCm Foundation | ROCm instance type, dependencies |
| 8-9 | ROCm Implementation | RCCL setup, model loading, generation |
| 10 | ROCm Testing | Multi-GPU tests, benchmarks |

---

## Testing Strategy

### Per-Backend Tests

**CPU**:
```bash
pytest tests/worker/engines/test_cpu_backend.py -v
python examples/cpu_inference_example.py
```

**WebGPU**:
```bash
# Requires browser environment
npm test  # Run Jest tests for web worker
python examples/webgpu_server_example.py  # Start server
```

**ROCm**:
```bash
pytest tests/worker/engines/test_rocm_backend.py -v --rocm
python examples/rocm_inference_example.py
```

### Integration Tests

```python
# tests/integration/test_multi_backend.py

def test_cpu_single_node():
    """Test CPU inference on single node."""
    cluster = start_test_cluster(backend="cpu", nodes=1)
    response = cluster.chat("Hello!")
    assert response.status_code == 200

def test_rocm_distributed():
    """Test ROCm distributed inference."""
    cluster = start_test_cluster(backend="rocm", nodes=2, gpus_per_node=1)
    response = cluster.chat("Explain quantum computing in 100 words")
    assert response.status_code == 200
```

---

## Deployment Guide

### CPU Deployment (Production Ready Week 3)

```bash
# Install exo with CPU backend
pip install exo[cpu]

# Start worker
exo worker --backend cpu --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Or with Docker
docker run -it exo:cpu --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### ROCm Deployment (Production Ready Week 10)

```bash
# Install exo with ROCm backend
pip install exo[rocm]

# Verify ROCm
rocm-smi

# Start worker
exo worker --backend rocm --model "meta-llama/Llama-2-7b-chat" --gpu 0

# Distributed (2 nodes)
# Node 1:
exo worker --backend rocm --distributed --rank 0 --world-size 2 --master-addr node1

# Node 2:
exo worker --backend rocm --distributed --rank 1 --world-size 2 --master-addr node1
```

---

## Success Metrics

| Backend | Metric | Target | Measurement |
|---------|--------|--------|-------------|
| **CPU** | Single-threaded throughput | 5-10 tok/s | TinyLlama-1.1B |
| **CPU** | Multi-threaded throughput | 15-20 tok/s | 8-core CPU |
| **CPU** | Memory efficiency | <4GB RAM | TinyLlama-1.1B Q4 |
| **WebGPU** | Browser throughput | 3-5 tok/s | TinyLlama in Chrome |
| **WebGPU** | Load time | <10s | Model load in browser |
| **ROCm** | Single GPU throughput | 40-50 tok/s | MI250X, Llama-2-7B |
| **ROCm** | Multi-GPU speedup | 1.8x | 2x MI250X, Llama-2-13B |
| **ROCm** | RCCL latency | <5ms | 2-node distributed |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **GGUF conversion complexity** | Medium | Use pre-converted models from TheBloke initially |
| **WebGPU browser compatibility** | Medium | Focus on Chromium-based browsers, make optional |
| **ROCm driver issues** | High | Test on AWS EC2 G4ad instances, provide Docker images |
| **Performance regression** | Low | Each backend independent, no risk to MLX |

---

## Next Steps

1. **Review this roadmap** with your team
2. **Set up development environment**:
   - Linux VM for CPU testing
   - AMD GPU access for ROCm (EC2 G4ad or local)
3. **Create GitHub issues** for each week's tasks
4. **Start Week 1: CPU Foundation** immediately

**Ready to begin implementation?** Start with Task 1.1: Engine Factory! 🚀
