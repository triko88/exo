# Cross-Platform Inference - Minimal Change Approach
## A Pragmatic Path to Multi-Backend Support

**Version:** 1.0
**Date:** 2026-01-15
**Status:** Alternative Design - Minimal Refactoring

---

## Executive Summary

This document presents a **minimal-change approach** to adding cross-platform support to exo. Instead of a comprehensive refactoring, this strategy leverages the existing `engines/` directory structure and implements new backends in parallel with minimal modifications to the core codebase.

**Key Principle**: Don't refactor what already works - add alongside it.

**Time Estimate**: 4-6 weeks for CUDA backend (vs 16 weeks for full abstraction)

---

## Current Architecture - Key Observation

The runner lifecycle only interacts with the inference engine through **3 functions**:

```python
# src/exo/worker/runner/runner.py (lines 42-47)

from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,      # Returns: Group | None
    load_mlx_items,      # Returns: (Model, TokenizerWrapper)
    mlx_force_oom,       # For testing only
)
```

This is a **remarkably narrow interface**! We can exploit this.

---

## Minimal Change Strategy

### Core Idea: Engine Module Pattern

Create parallel engine modules that expose the **exact same interface**:

```
src/exo/worker/engines/
├── mlx/                    # KEEP AS-IS (zero changes)
│   ├── utils_mlx.py
│   ├── generator/generate.py
│   └── auto_parallel.py
├── cuda/                   # NEW (parallel implementation)
│   ├── utils_cuda.py       # Same interface as utils_mlx.py
│   ├── generator/generate.py
│   └── auto_parallel.py
├── rocm/                   # NEW
│   └── ... (same structure)
├── cpu/                    # NEW
│   └── ... (same structure)
└── engine_factory.py       # NEW (5-line selector)
```

### Required Interface Contract

Each engine must implement:

```python
# engines/<backend>/utils_<backend>.py

def initialize_<backend>(bound_instance: BoundInstance) -> Group | None:
    """Initialize distributed group, return None for single-device."""
    pass

def load_<backend>_items(
    bound_instance: BoundInstance,
    group: Group | None
) -> tuple[Model, Tokenizer]:
    """Load model and tokenizer."""
    pass
```

```python
# engines/<backend>/generator/generate.py

def <backend>_generate(
    model: Model,
    tokenizer: Tokenizer,
    task: ChatCompletionTaskParams,
) -> Iterator[GenerationResponse]:
    """Stream generated tokens."""
    pass

def warmup_inference(model: Model, tokenizer: Tokenizer) -> int:
    """Warmup, return token count."""
    pass
```

---

## Implementation Plan

### Step 1: Add Engine Selection (1 day)

Create a simple factory:

```python
# src/exo/worker/engines/engine_factory.py

def get_engine_for_instance(instance: Instance):
    """Return (initialize_fn, load_fn, generate_fn, warmup_fn) for backend."""

    # Auto-detect from instance type or explicit backend field
    if isinstance(instance, MlxRingInstance | MlxJacclInstance):
        from exo.worker.engines.mlx.utils_mlx import initialize_mlx, load_mlx_items
        from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
        return initialize_mlx, load_mlx_items, mlx_generate, warmup_inference

    elif hasattr(instance, 'backend_name') and instance.backend_name == 'cuda':
        from exo.worker.engines.cuda.utils_cuda import initialize_cuda, load_cuda_items
        from exo.worker.engines.cuda.generator.generate import cuda_generate, warmup_inference
        return initialize_cuda, load_cuda_items, cuda_generate, warmup_inference

    # ... (similar for rocm, cpu)

    else:
        raise ValueError(f"Unknown instance type: {type(instance)}")
```

### Step 2: Modify Runner (5 lines changed)

```python
# src/exo/worker/runner/runner.py (MINIMAL CHANGE)

# OLD (lines 42-47):
# from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
# from exo.worker.engines.mlx.utils_mlx import initialize_mlx, load_mlx_items

# NEW (lines 42-44):
from exo.worker.engines.engine_factory import get_engine_for_instance

def main(bound_instance, event_sender, task_receiver):
    instance = bound_instance.instance

    # NEW: Get engine functions
    initialize_fn, load_fn, generate_fn, warmup_fn = get_engine_for_instance(instance)

    # Rest of the code UNCHANGED, just use the functions:
    # - initialize_fn() instead of initialize_mlx()
    # - load_fn() instead of load_mlx_items()
    # - generate_fn() instead of mlx_generate()
    # - warmup_fn() instead of warmup_inference()

    # ... existing code continues unchanged ...
```

**That's it for core changes!** The rest of runner.py stays identical.

### Step 3: Add Instance Type for CUDA (1 day)

```python
# src/exo/shared/types/worker/instances.py (ADDITION, not modification)

@dataclass
class CudaDistributedInstance(BaseInstance):
    """CUDA backend with NCCL distributed."""
    backend_name: str = "cuda"  # For engine_factory
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int

    # Optional: RDMA config
    use_rdma: bool = False
    ib_devices: list[str] | None = None

# Update union type
Instance = MlxRingInstance | MlxJacclInstance | CudaDistributedInstance  # Just add to union
```

### Step 4: Implement CUDA Engine (2-3 weeks)

Create parallel structure matching MLX:

```python
# src/exo/worker/engines/cuda/utils_cuda.py

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

def initialize_cuda(bound_instance: BoundInstance) -> torch.distributed.ProcessGroup | None:
    """Initialize NCCL distributed group."""
    instance = bound_instance.instance

    if not isinstance(instance, CudaDistributedInstance):
        return None  # Single GPU

    # Set up NCCL
    rank = get_rank_for_runner(bound_instance.bound_runner_id, instance)
    world_size = len(instance.shard_assignments.runner_to_shard)

    os.environ["MASTER_ADDR"] = instance.hosts_by_node[list(instance.hosts_by_node.keys())[0]][0].ipv4_address
    os.environ["MASTER_PORT"] = str(instance.ephemeral_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if instance.use_rdma:
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_NET_GDR_LEVEL"] = "5"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return dist.group.WORLD

def load_cuda_items(
    bound_instance: BoundInstance,
    group: torch.distributed.ProcessGroup | None
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load model with CUDA backend."""

    model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if group is None:
        # Single GPU - simple case
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        # Distributed - apply sharding
        model = load_sharded_cuda_model(
            model_path,
            bound_instance.bound_shard,
            group
        )

    return model, tokenizer

def load_sharded_cuda_model(model_path, shard_meta, group):
    """Apply tensor or pipeline parallelism."""

    if isinstance(shard_meta, TensorShardMetadata):
        # Use DeepSpeed or Accelerate for tensor parallel
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        # ... implementation ...

    elif isinstance(shard_meta, PipelineShardMetadata):
        # Load specific layers [start_layer:end_layer]
        # ... implementation ...

    return model
```

```python
# src/exo/worker/engines/cuda/generator/generate.py

from transformers import TextIteratorStreamer
import threading

def cuda_generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    task: ChatCompletionTaskParams,
) -> Iterator[GenerationResponse]:
    """Generate with CUDA backend using HuggingFace streaming."""

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        task.messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Set up streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # Generate in background thread
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "max_new_tokens": task.max_tokens or 2048,
        "temperature": task.temperature,
        "top_p": task.top_p,
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
            stats=GenerationStats(...)  # Track stats
        )

    thread.join()
    yield GenerationResponse(text="", token=token_count, finish_reason="stop", stats=...)

def warmup_inference(model, tokenizer) -> int:
    """Warmup CUDA inference."""
    warmup_prompt = "Hello"
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    return outputs.shape[1] - inputs.input_ids.shape[1]
```

### Step 5: Update Placement Algorithm (1 day)

```python
# src/exo/master/placement.py (MINIMAL ADDITION)

def place_instance(command: PlaceInstance, topology, current_instances):
    # ... existing cycle selection logic ...

    # NEW: Detect backend from node capabilities
    selected_backend = detect_backend_for_cycle(selected_cycle)

    # NEW: Create appropriate instance type
    if selected_backend == "cuda":
        instance = CudaDistributedInstance(
            instance_id=instance_id,
            shard_assignments=shard_assignments,
            hosts_by_node=get_cuda_hosts(selected_cycle),
            ephemeral_port=random_ephemeral_port(),
        )
    elif selected_backend == "mlx":
        # Existing MLX logic unchanged
        if command.instance_meta == InstanceMeta.MlxJaccl:
            instance = MlxJacclInstance(...)
        else:
            instance = MlxRingInstance(...)

    return {**target_instances, instance_id: instance}

def detect_backend_for_cycle(cycle: list[NodeInfo]) -> str:
    """Auto-detect best backend for cycle."""
    # Check first node's capabilities
    node = cycle[0]
    if node.node_profile and node.node_profile.devices:
        device = node.node_profile.devices[0]
        if device.device_type == DeviceType.CUDA:
            return "cuda"
        elif device.device_type == DeviceType.METAL:
            return "mlx"
    return "cpu"  # Fallback
```

---

## Comparison: Full Abstraction vs Minimal Approach

| Aspect | Full Abstraction (Original Design) | Minimal Approach (This Doc) |
|--------|-----------------------------------|----------------------------|
| **Time to First Backend** | 9 weeks (Phase 1-3) | 3-4 weeks |
| **Lines Changed in Core** | ~2000 LOC refactored | ~50 LOC changed |
| **Risk of Breaking MLX** | Medium (full refactor) | Near zero (MLX untouched) |
| **Code Duplication** | Low (shared abstractions) | Higher (parallel implementations) |
| **Extensibility** | Excellent (clean interfaces) | Good (copy-paste pattern) |
| **Testing Overhead** | High (validate all abstractions) | Low (only test new backend) |
| **Rollback Difficulty** | High | Trivial (just remove new files) |

---

## Migration Path

### Phase 1: Foundation (Week 1)
- Add `engine_factory.py`
- Modify `runner.py` (5 lines)
- Add `CudaDistributedInstance` type
- Verify MLX still works (zero regressions)

### Phase 2: CUDA Basics (Weeks 2-3)
- Implement `utils_cuda.py` (single GPU only)
- Implement `cuda_generate()` with HuggingFace
- Test single-GPU CUDA inference
- Benchmark against native PyTorch

### Phase 3: CUDA Distributed (Week 4)
- Add NCCL initialization in `initialize_cuda()`
- Implement tensor parallelism with Accelerate
- Implement pipeline parallelism with custom layer slicing
- Test 2-GPU distributed inference

### Phase 4: Production Ready (Weeks 5-6)
- Add error handling and logging
- Optimize memory usage
- Add performance profiling
- Documentation and examples

### Future: ROCm, CPU (Weeks 7+)
- Copy CUDA implementation, swap PyTorch ROCm
- Copy CUDA implementation, use llama.cpp for CPU
- Both follow the same pattern

---

## Key Advantages of This Approach

1. **MLX Code Untouched**: Zero risk of breaking existing functionality
2. **Fast to Market**: CUDA support in 4-6 weeks vs 16 weeks
3. **Easy Rollback**: Just delete new files if needed
4. **Incremental Testing**: Test each backend independently
5. **Clear Ownership**: Each backend is self-contained
6. **No Migration Pain**: Existing deployments unaffected

---

## Trade-offs and Limitations

### What You Give Up

1. **Code Duplication**: Each backend reimplements similar logic
2. **Inconsistent Patterns**: Backends may diverge in implementation
3. **Harder to Add Cross-Cutting Features**: Need to update all backends
4. **Less Type Safety**: No enforced interface contract (just convention)

### When to Migrate to Full Abstraction

Consider the full abstraction layer (original design) when:
- You have 4+ backends in production
- Code duplication becomes maintenance burden
- You need cross-backend features (e.g., automatic migration)
- Team size grows and needs strict contracts

**Migration Strategy**: Once you have MLX + CUDA + ROCm working with this minimal approach, you can **extract the common patterns** into the full abstraction layer. By then, you'll have real-world data on what the interface should be.

---

## Example: Complete CUDA Integration

Here's what changes in practice:

**Before (MLX only):**
```python
# runner.py
from exo.worker.engines.mlx.utils_mlx import initialize_mlx, load_mlx_items
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference

def main(bound_instance, event_sender, task_receiver):
    group = initialize_mlx(bound_instance)
    model, tokenizer = load_mlx_items(bound_instance, group)
    for response in mlx_generate(model, tokenizer, task):
        # ... yield responses
```

**After (MLX + CUDA):**
```python
# runner.py
from exo.worker.engines.engine_factory import get_engine_for_instance

def main(bound_instance, event_sender, task_receiver):
    initialize_fn, load_fn, generate_fn, warmup_fn = get_engine_for_instance(bound_instance.instance)

    group = initialize_fn(bound_instance)  # Calls initialize_mlx() or initialize_cuda()
    model, tokenizer = load_fn(bound_instance, group)
    for response in generate_fn(model, tokenizer, task):
        # ... yield responses
```

**Net Change**: 2 lines removed, 3 lines added = **5 line diff in runner.py**

---

## Decision Matrix: Which Approach?

| Your Situation | Recommended Approach |
|----------------|---------------------|
| Need CUDA support ASAP for production | **Minimal Approach** |
| Planning to support 5+ backends | Full Abstraction |
| Small team (1-3 devs) | **Minimal Approach** |
| Large team with strict architecture standards | Full Abstraction |
| Tight deadline (< 2 months) | **Minimal Approach** |
| Building long-term platform | Full Abstraction |
| Risk-averse deployment | **Minimal Approach** |
| Need perfect code architecture | Full Abstraction |

---

## Recommendation

**Start with the Minimal Approach**, then migrate to Full Abstraction:

1. **Weeks 1-6**: Implement CUDA using minimal approach
2. **Week 7**: Deploy to production, gather feedback
3. **Weeks 8-10**: Add ROCm and CPU backends (same pattern)
4. **Week 11**: Evaluate code duplication pain points
5. **Weeks 12-16**: Extract common abstractions (if needed)

This gives you:
- ✅ Working CUDA support in 1.5 months (vs 4 months)
- ✅ Production validation before committing to architecture
- ✅ Real-world data to inform abstraction design
- ✅ Zero risk to existing MLX deployments

---

## Conclusion

The minimal-change approach prioritizes **speed and safety** over **perfect architecture**. It's ideal for teams that need cross-platform support quickly while minimizing risk.

The full abstraction layer (original design document) is still valuable as the **long-term target**. But by starting minimal, you:
1. Validate the business case faster
2. Learn what the right abstractions are from real usage
3. Keep options open (can always refactor later)
4. Ship value to users sooner

**Recommendation**: Use this minimal approach to get CUDA working, then revisit the full abstraction design in 3-6 months with production learnings.

---

**Document History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-15 | Claude (AI Assistant) | Initial minimal-change design |
