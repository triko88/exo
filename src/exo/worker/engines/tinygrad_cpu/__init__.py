from typing import Any, Protocol, runtime_checkable
from tinygrad.tensor import Tensor
from tinygrad.device import Device

"""
    This class is the wrapper around the exo's infrastructure to support
    tinygrad CPU inference. By default, tinygrad selects the best available
    inference backend. However in non-Apple silicon systems, the operating
    system treats RAM and VRAM differently. This includes Intel Macs, 
    however this implementation doesn't support those.

    Mainstream consumer grade x86 (Intel/AMD) processors come with an
    integrated GPU, but the firmware selects a portion of the RAM that the
    operating system would consider as VRAM on boot.

    Forcing the node to select CPU or GPU for inference on such hardware
    ensures scheduler-runtime stability, and consistent memory availablity.
    Which is critical for loading sharded models across multiple instances.

    Both integrated and disrete GPU will be handled when Linux GPU inference
    will be implemented in the near future.
"""

Device.DEFAULT = "CPU"

@runtime_checkable
class Model(Protocol):
    def __call__(
        self,
        x: Tensor,
        input_embeddings: Tensor,
    ) -> Tensor: ...

@runtime_checkable
class Detokenizer(Protocol):
    def reset(self) -> None: ...
    def add(self, token: int) -> None: ...
    def get(self) -> str: ...

    @property
    def last_segment(self) -> str: ...

@runtime_checkable
class TokenizerWrapper(Protocol):
    bos_token: str | None = None
    eos_token_ids: list[int] | None = None
    detokenizer: Detokenizer 

    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
    ) -> list[int]: ...

    def apply_message_template(
        self,
        message_dicts: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str: ...
