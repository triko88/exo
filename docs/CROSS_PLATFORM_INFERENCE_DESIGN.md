# Cross-Platform Inference Architecture Design Document
## Enabling Linux Support with CPU, ROCm, CUDA, and WebGPU Backends

**Version:** 1.0
**Date:** 2026-01-15
**Status:** Design Phase

---

## Executive Summary

The **exo** distributed AI inference platform currently relies exclusively on Apple's MLX framework, limiting deployment to macOS devices with Apple Silicon. This design document outlines a comprehensive architectural refactoring to enable cross-platform support for Linux systems with multiple inference backends: **CPU, ROCm (AMD), CUDA (NVIDIA), and WebGPU**.

The proposed architecture introduces a **Backend Abstraction Layer** that decouples inference execution from the orchestration logic while preserving the existing distributed inference capabilities (tensor/pipeline parallelism, RDMA networking, topology-aware placement).

**Key Goals:**
- Enable Linux support alongside macOS
- Support multiple inference backends: MLX (Apple), CUDA (NVIDIA), ROCm (AMD), CPU-only, WebGPU
- Maintain performance characteristics of existing MLX implementation
- Preserve distributed inference capabilities across heterogeneous device clusters
- Minimize breaking changes to existing APIs

---

## 1. Current Architecture Analysis

### 1.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Node (Orchestrator)                │
│  ├─ Placement Engine (placement.py)                          │
│  ├─ REST API Server (api.py)                                 │
│  └─ State Management (shared/types/state.py)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │   libp2p Networking       │
                │   (Rust + PyO3 Bindings)  │
                └─────────────┬─────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐      ┌──────▼───────┐     ┌──────▼───────┐
│ Worker Node 1│      │ Worker Node 2│     │ Worker Node N│
│ ├─ Runner    │      │ ├─ Runner    │     │ ├─ Runner    │
│ ├─ MLX Engine│      │ ├─ MLX Engine│     │ ├─ MLX Engine│
│ └─ Model     │      │ └─ Model     │     │ └─ Model     │
└──────────────┘      └──────────────┘     └──────────────┘
```

### 1.2 Apple-Centric Dependencies

| Component | Apple-Specific Code | Location |
|-----------|---------------------|----------|
| **Inference Engine** | `mlx`, `mlx-lm` | `worker/engines/mlx/` |
| **Distributed Backend** | MLX Ring (TCP), MLX JACCL (RDMA/Thunderbolt) | `worker/engines/mlx/utils_mlx.py:101-161` |
| **Model Loading** | `mlx_lm.load_model()` | `worker/engines/mlx/utils_mlx.py:176-253` |
| **Generation** | `mlx_lm.stream_generate()` | `worker/engines/mlx/generator/generate.py:118-193` |
| **Device Management** | MLX Metal, ANE detection | `shared/system_info.py` |
| **System Monitoring** | `macmon` binary | `worker/utils/macmon.py` |
| **Parallelization** | `tensor_auto_parallel`, `pipeline_auto_parallel` | `worker/engines/mlx/auto_parallel.py` |

### 1.3 Critical Observations

1. **Tight Coupling**: Inference logic is tightly coupled to MLX throughout the runner lifecycle (`runner/runner.py`)
2. **Instance Types**: `MlxRingInstance` and `MlxJacclInstance` are backend-specific but used in core orchestration
3. **Distributed Init**: MLX distributed initialization is directly invoked in the worker setup path
4. **Type System**: Pydantic types assume MLX-specific metadata (e.g., `PipelineShardMetadata`, `TensorShardMetadata`)
5. **Platform Checks**: `sys.platform == "darwin"` scattered throughout codebase

---

## 2. Goals and Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Support Linux (Ubuntu 22.04+, Debian 12+) alongside macOS | **P0** |
| FR-2 | Implement CUDA backend for NVIDIA GPUs (Compute Capability ≥7.0) | **P0** |
| FR-3 | Implement ROCm backend for AMD GPUs (RDNA2+, CDNA) | **P1** |
| FR-4 | Implement CPU-only backend using optimized libraries | **P0** |
| FR-5 | Implement WebGPU backend for browser/edge deployment | **P2** |
| FR-6 | Support heterogeneous clusters (mixed Apple/NVIDIA/AMD/CPU) | **P1** |
| FR-7 | Maintain tensor and pipeline parallelism capabilities | **P0** |
| FR-8 | Preserve existing REST API contract | **P0** |
| FR-9 | Enable per-backend configuration (quantization, precision, memory limits) | **P1** |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Performance: ≥90% of native backend throughput | **P0** |
| NFR-2 | Memory Overhead: ≤10% additional memory per backend abstraction | **P0** |
| NFR-3 | Latency: Inter-device communication latency ≤5ms overhead vs MLX JACCL | **P1** |
| NFR-4 | Compatibility: Support models from HuggingFace Hub without conversion | **P0** |
| NFR-5 | Extensibility: New backend integration ≤500 LOC | **P1** |

### 2.3 Out of Scope

- Windows support (future consideration)
- TPU backends
- Custom ASIC support (Groq, Cerebras)
- Model format conversion pipelines
- Training/fine-tuning workloads

---

## 3. Proposed Architecture

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER (Unchanged)                 │
│  ├─ Master (Placement, API, State)                               │
│  ├─ Worker (Runner Lifecycle, Task Handling)                     │
│  └─ Networking (libp2p, DHT, Keep-Alive)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│              BACKEND ABSTRACTION LAYER (NEW)                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Abstract Interfaces                                        │ │
│  │  ├─ BaseInferenceBackend                                    │ │
│  │  ├─ BaseDistributedGroup                                    │ │
│  │  ├─ BaseDeviceCapability                                    │ │
│  │  └─ BaseModelLoader                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Unified APIs                                               │ │
│  │  ├─ generate(model, prompt, params) → Iterator[Chunk]      │ │
│  │  ├─ load_model(path, shard_meta) → Model                   │ │
│  │  ├─ init_distributed(config) → Group                       │ │
│  │  └─ detect_devices() → List[DeviceCapability]              │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬─────────────┐
        │                  │                  │             │
┌───────▼──────┐   ┌──────▼──────┐   ┌──────▼─────┐  ┌────▼────┐
│ MLX Backend  │   │CUDA Backend │   │ROCm Backend│  │CPU/WebGPU│
│ (Apple GPU)  │   │(NVIDIA GPU) │   │ (AMD GPU)  │  │  Backends│
│              │   │             │   │            │  │          │
│ • Metal      │   │ • NCCL      │   │ • RCCL     │  │• llama.cpp│
│ • MLX Ring   │   │ • cuDNN     │   │ • MIOpen   │  │• ONNX RT │
│ • MLX JACCL  │   │ • TensorRT  │   │ • hipBLAS  │  │• WebGPU  │
└──────────────┘   └─────────────┘   └────────────┘  └──────────┘
```

### 3.2 Layer Responsibilities

#### **Orchestration Layer** (Existing, Minimal Changes)
- Cluster topology management
- Device discovery and ranking
- Placement decisions (cycle selection, memory checks)
- Task routing and lifecycle management
- REST API surface

#### **Backend Abstraction Layer** (New)
- Unified interface for model loading, inference, and distributed communication
- Backend registration and capability negotiation
- Automatic backend selection based on device detection
- Cross-backend type conversion

#### **Backend Implementations** (New + Adapted MLX)
- Backend-specific inference engines
- Distributed primitives (all-reduce, broadcast, barrier)
- Memory management and quantization
- Performance profiling hooks

---

## 4. Backend Abstraction Layer Design

### 4.1 Core Abstractions

#### 4.1.1 BaseInferenceBackend

```python
# src/exo/worker/engines/base_backend.py

from abc import ABC, abstractmethod
from typing import Iterator, Any
from exo.shared.types.worker.shards import ShardMetadata
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse

class BaseInferenceBackend(ABC):
    """Abstract base class for all inference backends."""

    @abstractmethod
    def backend_name(self) -> str:
        """Returns unique backend identifier (e.g., 'mlx', 'cuda', 'rocm')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend can run on current system."""
        pass

    @abstractmethod
    def load_model(
        self,
        model_path: Path,
        shard_meta: ShardMetadata,
        group: BaseDistributedGroup | None = None
    ) -> tuple[Any, Any]:  # (model, tokenizer)
        """Load model weights and tokenizer."""
        pass

    @abstractmethod
    def generate(
        self,
        model: Any,
        tokenizer: Any,
        task: ChatCompletionTaskParams,
    ) -> Iterator[GenerationResponse]:
        """Stream generated tokens."""
        pass

    @abstractmethod
    def cleanup(self, model: Any, tokenizer: Any, group: Any | None) -> None:
        """Release resources."""
        pass

    @abstractmethod
    def warmup(self, model: Any, tokenizer: Any) -> int:
        """Warmup inference engine, return tokens generated."""
        pass
```

#### 4.1.2 BaseDistributedGroup

```python
# src/exo/worker/engines/base_distributed.py

from abc import ABC, abstractmethod
from typing import Any

class BaseDistributedGroup(ABC):
    """Abstract distributed communication group."""

    @abstractmethod
    def rank(self) -> int:
        """Returns rank of current process in group."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Returns total number of processes in group."""
        pass

    @abstractmethod
    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        """All-reduce collective operation."""
        pass

    @abstractmethod
    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        """Broadcast tensor from src rank to all ranks."""
        pass

    @abstractmethod
    def barrier(self) -> None:
        """Synchronization barrier."""
        pass

    @abstractmethod
    def send(self, tensor: Any, dst: int, tag: int = 0) -> None:
        """Point-to-point send (for pipeline parallelism)."""
        pass

    @abstractmethod
    def recv(self, shape: tuple, dtype: Any, src: int, tag: int = 0) -> Any:
        """Point-to-point receive (for pipeline parallelism)."""
        pass
```

#### 4.1.3 BaseDeviceCapability

```python
# src/exo/worker/engines/base_device.py

from enum import Enum
from pydantic import BaseModel
from exo.shared.types.memory import Memory

class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"      # NVIDIA GPU
    ROCM = "rocm"      # AMD GPU
    METAL = "metal"    # Apple GPU
    WEBGPU = "webgpu"  # Browser/WebGPU

class DeviceCapability(BaseModel):
    """Describes hardware capabilities."""
    device_type: DeviceType
    device_index: int  # 0, 1, 2 for multi-GPU
    device_name: str   # "NVIDIA RTX 4090", "AMD MI300X"
    compute_capability: str | None  # "8.6" for CUDA, "gfx1100" for ROCm
    memory_total: Memory
    memory_available: Memory
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    supports_int4: bool
    backend_name: str  # Maps to BaseInferenceBackend.backend_name()
    rdma_capable: bool  # For GPUDirect RDMA (CUDA/ROCm)
    rdma_device_name: str | None  # "mlx5_0" for InfiniBand
```

### 4.2 Backend Registry

```python
# src/exo/worker/engines/registry.py

from typing import Type
from exo.worker.engines.base_backend import BaseInferenceBackend

class BackendRegistry:
    """Global registry for inference backends."""

    _backends: dict[str, Type[BaseInferenceBackend]] = {}

    @classmethod
    def register(cls, backend_class: Type[BaseInferenceBackend]) -> None:
        """Register a backend implementation."""
        name = backend_class().backend_name()
        cls._backends[name] = backend_class

    @classmethod
    def get_backend(cls, name: str) -> Type[BaseInferenceBackend]:
        """Retrieve backend by name."""
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' not registered")
        return cls._backends[name]

    @classmethod
    def detect_available_backends(cls) -> list[str]:
        """Return list of available backend names."""
        return [
            name for name, backend_cls in cls._backends.items()
            if backend_cls().is_available()
        ]

    @classmethod
    def auto_select_backend(cls, device: DeviceCapability) -> str:
        """Automatically select best backend for device."""
        # Mapping: DeviceType → Backend Name
        mapping = {
            DeviceType.METAL: "mlx",
            DeviceType.CUDA: "cuda",
            DeviceType.ROCM: "rocm",
            DeviceType.CPU: "cpu",
            DeviceType.WEBGPU: "webgpu",
        }
        backend_name = mapping.get(device.device_type)
        if backend_name and backend_name in cls._backends:
            return backend_name
        # Fallback to CPU
        return "cpu"
```

### 4.3 Backend Implementations

#### 4.3.1 MLX Backend (Adapted from existing)

```python
# src/exo/worker/engines/mlx_backend.py

from exo.worker.engines.base_backend import BaseInferenceBackend
from exo.worker.engines.mlx.utils_mlx import load_mlx_items, mlx_cleanup
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference

class MLXBackend(BaseInferenceBackend):
    def backend_name(self) -> str:
        return "mlx"

    def is_available(self) -> bool:
        try:
            import mlx.core as mx
            return mx.metal.is_available()
        except ImportError:
            return False

    def load_model(self, model_path, shard_meta, group=None):
        # Delegates to existing load_mlx_items()
        return load_mlx_items(bound_instance, group)

    def generate(self, model, tokenizer, task):
        # Delegates to existing mlx_generate()
        return mlx_generate(model, tokenizer, task)

    def cleanup(self, model, tokenizer, group):
        mlx_cleanup(model, tokenizer, group)

    def warmup(self, model, tokenizer):
        return warmup_inference(model, tokenizer)

# Register backend
BackendRegistry.register(MLXBackend)
```

#### 4.3.2 CUDA Backend (New Implementation)

```python
# src/exo/worker/engines/cuda_backend.py

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

class CUDABackend(BaseInferenceBackend):
    def backend_name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def load_model(self, model_path, shard_meta, group=None):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if group is None:
            # Single GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Distributed: Use DeepSpeed, Accelerate, or custom sharding
            model = load_sharded_model_cuda(model_path, shard_meta, group)

        return model, tokenizer

    def generate(self, model, tokenizer, task):
        # Use HuggingFace TextIteratorStreamer for streaming
        from transformers import TextIteratorStreamer
        # Implementation details...

    # ... (other methods)

BackendRegistry.register(CUDABackend)
```

#### 4.3.3 ROCm Backend (New Implementation)

```python
# src/exo/worker/engines/rocm_backend.py

# Similar to CUDA backend but with ROCm-specific optimizations
# Uses PyTorch with ROCm backend, RCCL for distributed

class ROCmBackend(BaseInferenceBackend):
    def backend_name(self) -> str:
        return "rocm"

    def is_available(self) -> bool:
        return torch.cuda.is_available() and "rocm" in torch.version.hip

    # ... (similar implementation to CUDA)

BackendRegistry.register(ROCmBackend)
```

#### 4.3.4 CPU Backend (New Implementation)

```python
# src/exo/worker/engines/cpu_backend.py

# Uses llama.cpp Python bindings for optimized CPU inference

from llama_cpp import Llama

class CPUBackend(BaseInferenceBackend):
    def backend_name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True  # Always available

    def load_model(self, model_path, shard_meta, group=None):
        # Use GGUF quantized models for efficiency
        model = Llama(
            model_path=str(model_path / "model.gguf"),
            n_ctx=4096,
            n_threads=os.cpu_count(),
            use_mlock=True,
        )
        tokenizer = model  # llama.cpp handles tokenization
        return model, tokenizer

    # ... (other methods)

BackendRegistry.register(CPUBackend)
```

---

## 5. Device Detection and Capability Management

### 5.1 Unified Device Discovery

```python
# src/exo/worker/utils/device_detection.py

def detect_devices() -> list[DeviceCapability]:
    """Detect all available compute devices on the system."""
    devices = []

    # Apple Metal (macOS)
    if sys.platform == "darwin":
        devices.extend(_detect_metal_devices())

    # NVIDIA CUDA (Linux/macOS)
    if torch.cuda.is_available():
        if "rocm" in torch.version.hip:
            devices.extend(_detect_rocm_devices())
        else:
            devices.extend(_detect_cuda_devices())

    # CPU (always available)
    devices.append(_detect_cpu_device())

    # WebGPU (if available)
    if _is_webgpu_available():
        devices.extend(_detect_webgpu_devices())

    return devices

def _detect_cuda_devices() -> list[DeviceCapability]:
    """Detect NVIDIA GPUs."""
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append(DeviceCapability(
            device_type=DeviceType.CUDA,
            device_index=i,
            device_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            memory_total=Memory.from_bytes(props.total_memory),
            memory_available=Memory.from_bytes(
                props.total_memory - torch.cuda.memory_allocated(i)
            ),
            supports_fp16=props.major >= 7,
            supports_bf16=props.major >= 8,
            supports_int8=True,
            supports_int4=True,
            backend_name="cuda",
            rdma_capable=_check_gpudirect_rdma(),
            rdma_device_name=_get_ib_device_for_gpu(i),
        ))
    return devices
```

### 5.2 Integration with Topology

Current `NodePerformanceProfile` needs extension:

```python
# src/exo/shared/types/topology.py (modifications)

class NodePerformanceProfile(BaseModel):
    # ... existing fields ...
    devices: list[DeviceCapability]  # NEW: List all devices on this node
    preferred_backend: str  # NEW: "mlx", "cuda", "rocm", "cpu"
```

Master placement algorithm should consider device capabilities:

```python
# src/exo/master/placement.py (modifications)

def filter_cycles_by_device_compatibility(
    cycles: list[list[NodeInfo]],
    required_backend: str | None = None
) -> list[list[NodeInfo]]:
    """Filter cycles where all nodes support the same backend."""
    compatible_cycles = []
    for cycle in cycles:
        # Check if all nodes in cycle support a common backend
        common_backends = set.intersection(*[
            set(node.node_profile.devices[0].backend_name)
            for node in cycle
            if node.node_profile
        ])
        if common_backends:
            compatible_cycles.append(cycle)
    return compatible_cycles
```

---

## 6. Distributed Communication Layer

### 6.1 Communication Backends

Each inference backend requires a corresponding distributed communication backend:

| Inference Backend | Communication Backend | Protocol |
|-------------------|----------------------|----------|
| MLX | MLX Ring | TCP sockets |
| MLX | MLX JACCL | RDMA over Thunderbolt/InfiniBand |
| CUDA | NCCL | GPUDirect RDMA / TCP |
| ROCm | RCCL | ROCm RDMA / TCP |
| CPU | Gloo | TCP sockets |
| WebGPU | WebRTC | Browser P2P |

### 6.2 Unified Instance Types

Replace `MlxRingInstance`, `MlxJacclInstance` with generic types:

```python
# src/exo/shared/types/worker/instances.py (refactored)

class DistributedBackend(str, Enum):
    MlxRing = "mlx_ring"
    MlxJaccl = "mlx_jaccl"
    NCCL = "nccl"
    RCCL = "rccl"
    Gloo = "gloo"
    WebRTC = "webrtc"

class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments
    backend_name: str  # NEW: "mlx", "cuda", "rocm", etc.
    distributed_backend: DistributedBackend  # NEW

class TCPDistributedInstance(BaseInstance):
    """Generic TCP-based distributed instance (MLX Ring, Gloo, NCCL/TCP)."""
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int

class RDMADistributedInstance(BaseInstance):
    """RDMA-based distributed instance (JACCL, NCCL/IB, RCCL/IB)."""
    rdma_devices: list[list[str | None]]  # Generic IB device names
    coordinators: dict[NodeId, str]

Instance = TCPDistributedInstance | RDMADistributedInstance
```

### 6.3 Backend-Specific Initialization

```python
# src/exo/worker/engines/cuda_backend.py (distributed init)

class CUDADistributedGroup(BaseDistributedGroup):
    """NCCL-based distributed group for CUDA."""

    def __init__(self, instance: Instance, rank: int):
        if isinstance(instance, TCPDistributedInstance):
            # Use NCCL over TCP
            os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Or auto-detect
            torch.distributed.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=len(instance.shard_assignments.runner_to_shard),
                init_method=f"tcp://{instance.coordinators[0]}:{instance.ephemeral_port}"
            )
        elif isinstance(instance, RDMADistributedInstance):
            # Use NCCL with GPUDirect RDMA
            os.environ["NCCL_IB_DISABLE"] = "0"
            os.environ["NCCL_NET_GDR_LEVEL"] = "5"
            # ... (RDMA-specific setup)

        self._group = torch.distributed.group.WORLD

    def rank(self) -> int:
        return torch.distributed.get_rank()

    def all_reduce(self, tensor, op="sum"):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor

    # ... (other methods)
```

---

## 7. Model Loading and Execution Flow

### 7.1 Refactored Runner Lifecycle

```python
# src/exo/worker/runner/runner.py (refactored)

from exo.worker.engines.registry import BackendRegistry

def main(bound_instance, event_sender, task_receiver):
    # Detect backend from instance metadata
    backend_name = bound_instance.instance.backend_name
    backend_class = BackendRegistry.get_backend(backend_name)
    backend = backend_class()

    model = None
    tokenizer = None
    group = None

    current_status = RunnerIdle()

    for task in tasks:
        match task:
            case ConnectToGroup() if isinstance(current_status, RunnerIdle):
                current_status = RunnerConnecting()
                # Backend-specific distributed init
                group = backend.init_distributed(bound_instance)
                current_status = RunnerConnected()

            case LoadModel() if isinstance(current_status, RunnerConnected):
                current_status = RunnerLoading()
                model, tokenizer = backend.load_model(
                    model_path=build_model_path(...),
                    shard_meta=bound_instance.bound_shard,
                    group=group
                )
                current_status = RunnerLoaded()

            case StartWarmup() if isinstance(current_status, RunnerLoaded):
                current_status = RunnerWarmingUp()
                backend.warmup(model, tokenizer)
                current_status = RunnerReady()

            case ChatCompletion(task_params=params) if isinstance(current_status, RunnerReady):
                current_status = RunnerRunning()
                for response in backend.generate(model, tokenizer, params):
                    event_sender.send(ChunkGenerated(...))
                current_status = RunnerReady()

            case Shutdown():
                backend.cleanup(model, tokenizer, group)
                break
```

### 7.2 Model Format Compatibility

Different backends may require different model formats:

| Backend | Preferred Format | Conversion Strategy |
|---------|-----------------|---------------------|
| MLX | MLX format (`.npz`, `.safetensors`) | `mlx_lm.convert` from HF |
| CUDA/ROCm | HuggingFace (`.safetensors`) | Direct load with `transformers` |
| CPU | GGUF (quantized) | `llama.cpp` conversion tools |
| WebGPU | ONNX | Export via `optimum` |

**Strategy**:
- Store models in HuggingFace format as the canonical source
- Perform lazy conversion on first load per backend
- Cache converted models in `~/.cache/exo/<model_id>/<backend_name>/`

---

## 8. Migration Strategy and Phases

### Phase 1: Foundation (Weeks 1-3)

**Goal**: Establish abstraction layer without breaking existing functionality

**Tasks**:
1. Create `BaseInferenceBackend`, `BaseDistributedGroup`, `BaseDeviceCapability` interfaces
2. Implement `BackendRegistry` and device detection framework
3. Refactor MLX code into `MLXBackend` class (no logic changes)
4. Update `runner.py` to use backend abstraction
5. Add `backend_name` field to `Instance` types (default: "mlx")
6. **Testing**: Ensure all existing MLX tests pass with new abstraction

**Deliverables**:
- `/src/exo/worker/engines/base_backend.py`
- `/src/exo/worker/engines/mlx_backend.py` (refactored)
- `/src/exo/worker/engines/registry.py`
- Unit tests for registry and MLX backend

---

### Phase 2: CPU Backend (Weeks 4-5)

**Goal**: Enable CPU-only inference for Linux systems

**Tasks**:
1. Implement `CPUBackend` using `llama.cpp` Python bindings
2. Add CPU device detection for Linux
3. Implement `CPUDistributedGroup` using Gloo (PyTorch)
4. Add model conversion pipeline for GGUF format
5. Update placement algorithm to handle CPU-only nodes
6. **Testing**: Deploy 2-node CPU cluster on Linux VMs

**Deliverables**:
- `/src/exo/worker/engines/cpu_backend.py`
- CPU device detection in `device_detection.py`
- Integration tests for CPU inference

---

### Phase 3: CUDA Backend (Weeks 6-9)

**Goal**: Enable NVIDIA GPU support with NCCL distributed

**Tasks**:
1. Implement `CUDABackend` using HuggingFace `transformers`
2. Add CUDA device detection (GPU properties, compute capability)
3. Implement `CUDADistributedGroup` with NCCL backend
4. Support tensor parallelism via DeepSpeed or Accelerate
5. Support pipeline parallelism with custom send/recv ops
6. Enable GPUDirect RDMA for InfiniBand clusters
7. **Testing**: Deploy 4-GPU cluster (2x RTX 4090 nodes)

**Deliverables**:
- `/src/exo/worker/engines/cuda_backend.py`
- NCCL distributed group implementation
- Tensor/pipeline parallelism for CUDA
- Performance benchmarks vs MLX

---

### Phase 4: ROCm Backend (Weeks 10-12)

**Goal**: Enable AMD GPU support with RCCL distributed

**Tasks**:
1. Implement `ROCmBackend` (fork of CUDA backend)
2. Add ROCm device detection (HIP version, gfx architecture)
3. Implement `ROCmDistributedGroup` with RCCL backend
4. Handle ROCm-specific quirks (e.g., MIOpen tuning)
5. **Testing**: Deploy on AMD MI200 or RDNA3 GPUs

**Deliverables**:
- `/src/exo/worker/engines/rocm_backend.py`
- ROCm-specific documentation

---

### Phase 5: Heterogeneous Clusters (Weeks 13-14)

**Goal**: Enable mixed Apple/NVIDIA/AMD/CPU clusters

**Tasks**:
1. Update placement algorithm to handle heterogeneous cycles
2. Implement cross-backend communication (MLX ↔ CUDA)
3. Handle precision mismatches (FP16 vs BF16)
4. Add backend affinity hints to topology
5. **Testing**: Deploy mixed cluster (Mac Mini + Linux + NVIDIA GPU)

**Deliverables**:
- Heterogeneous placement algorithm
- Cross-backend communication tests

---

### Phase 6: WebGPU Backend (Weeks 15-16) [Optional]

**Goal**: Enable browser-based inference nodes

**Tasks**:
1. Implement `WebGPUBackend` using ONNX Runtime Web
2. Add WebRTC-based distributed communication
3. Deploy web worker for background inference
4. **Testing**: Browser-based node joining cluster

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/worker/engines/test_backend_registry.py

def test_register_backend():
    class DummyBackend(BaseInferenceBackend):
        def backend_name(self): return "dummy"
        # ... (minimal implementation)

    BackendRegistry.register(DummyBackend)
    assert "dummy" in BackendRegistry._backends

def test_auto_select_backend():
    device = DeviceCapability(device_type=DeviceType.CUDA, ...)
    backend = BackendRegistry.auto_select_backend(device)
    assert backend == "cuda"
```

### 9.2 Integration Tests

```python
# tests/integration/test_cuda_backend.py

@pytest.mark.requires_cuda
def test_cuda_single_gpu_inference():
    backend = CUDABackend()
    model, tokenizer = backend.load_model(...)
    responses = list(backend.generate(model, tokenizer, task))
    assert len(responses) > 0
    assert responses[-1].finish_reason == "stop"

@pytest.mark.requires_multi_gpu
def test_cuda_tensor_parallelism():
    # Test 2-GPU tensor parallel inference
    ...
```

### 9.3 End-to-End Tests

```python
# tests/e2e/test_heterogeneous_cluster.py

def test_mixed_mlx_cuda_cluster():
    """Deploy cluster with 1 Mac (MLX) + 1 Linux (CUDA) node."""
    cluster = start_test_cluster([
        Node(platform="darwin", backend="mlx"),
        Node(platform="linux", backend="cuda"),
    ])
    response = cluster.chat_completion("Explain quantum computing")
    assert response.status_code == 200
```

### 9.4 Performance Benchmarks

| Metric | Baseline (MLX) | Target (CUDA/ROCm) |
|--------|----------------|---------------------|
| **Tokens/sec (single GPU)** | 50 tok/s | ≥45 tok/s (90%) |
| **Latency (first token)** | 200ms | ≤220ms |
| **Memory overhead** | 16GB model → 16GB used | ≤17.6GB (10% overhead) |
| **Distributed throughput (2-GPU tensor)** | 80 tok/s | ≥72 tok/s |

---

## 10. Performance Considerations

### 10.1 Memory Management

- **MLX**: Unified memory (shared CPU/GPU)
- **CUDA**: Separate device memory, requires explicit transfers
- **ROCm**: Similar to CUDA
- **CPU**: System RAM only

**Strategy**: Each backend implements custom memory pooling to minimize allocations during generation.

### 10.2 Kernel Fusion

- **MLX**: Automatic kernel fusion via lazy evaluation
- **CUDA**: Use Flash Attention 2, custom fused kernels
- **ROCm**: Use Composable Kernel library

### 10.3 Quantization

| Backend | Supported Quantization |
|---------|------------------------|
| MLX | 4-bit, 8-bit (custom) |
| CUDA | INT8 (TensorRT), INT4 (GPTQ/AWQ) |
| ROCm | INT8 (MIGraphX) |
| CPU | Q4_K_M, Q5_K_M (llama.cpp GGUF) |

**Strategy**: Auto-detect and apply optimal quantization per backend during model loading.

---

## 11. Risk Mitigation

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| **Performance regression on MLX** | High | Phase 1 refactoring must maintain 100% performance parity. Add benchmark CI. |
| **NCCL/RCCL setup complexity** | Medium | Provide Docker containers with pre-configured environments. |
| **Model format conversion failures** | Medium | Graceful fallback to HuggingFace format. Cache conversions. |
| **Cross-backend communication overhead** | Medium | Limit heterogeneous clusters to pipeline parallelism (less inter-device traffic). |
| **Limited ROCm testing hardware** | Medium | Partner with AMD for MI200 access, or use cloud (AWS EC2 p4d). |
| **Dependency conflicts (PyTorch ROCm vs CUDA)** | High | Use separate virtual environments or Docker containers per backend. |

---

## 12. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Platform Coverage** | Linux + macOS support | CI passes on Ubuntu 22.04 + macOS 14+ |
| **Backend Coverage** | MLX, CUDA, ROCm, CPU functional | All backends pass integration tests |
| **Performance Parity** | ≥90% of native backend throughput | tok/s benchmarks vs baseline |
| **API Stability** | Zero breaking changes to REST API | API contract tests |
| **Community Adoption** | 50+ GitHub stars on Linux support | GitHub analytics |

---

## 13. Future Enhancements (Beyond Initial Scope)

1. **Dynamic Backend Switching**: Switch backends mid-execution based on load
2. **Automatic Model Sharding**: Use Alpa/FlexFlow for optimal partitioning
3. **Multi-Tier Memory**: Offload inactive layers to CPU/NVMe (FlexGen-style)
4. **Custom CUDA Kernels**: Implement exo-specific optimizations (e.g., sparse attention)
5. **Distributed Training**: Extend beyond inference to support fine-tuning
6. **Edge TPU Support**: Add Coral/EdgeTPU backend for embedded devices

---

## 14. Conclusion

This design document outlines a comprehensive path to transforming **exo** from an Apple-centric distributed inference platform into a truly cross-platform solution supporting Linux with NVIDIA, AMD, and CPU backends. The proposed **Backend Abstraction Layer** decouples inference execution from orchestration logic, enabling extensibility while preserving the system's core strengths: topology-aware placement, RDMA networking, and efficient parallelism.

**Key Takeaways**:
- Phased migration minimizes risk and maintains backwards compatibility
- Abstract interfaces enable future backend additions with minimal effort
- Heterogeneous cluster support unlocks new deployment scenarios
- Performance targets ensure the abstraction doesn't compromise efficiency

**Next Steps**:
1. Review and approve this design document
2. Create GitHub issues for Phase 1 tasks
3. Set up CI environments for Linux testing
4. Begin implementation of `BaseInferenceBackend` interface

---

**Document History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-15 | Claude (AI Assistant) | Initial design document |

---

**License**: This design document is part of the exo project and follows the same license terms.

**Contributors**: This document was created in collaboration with the exo development community.
