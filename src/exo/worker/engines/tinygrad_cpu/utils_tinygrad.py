from typing import Callable, Tuple
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.utils import load_state_dict, safe_load

from transformers import AutoTokenizer
from transformers.models.distilbert.modeling_distilbert import Transformer
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.tinygrad_cpu import Model, TokenizerWrapper
from exo.shared.types.worker.instances import BoundInstance

Device.DEFAULT = "CPU"


def initialize_tinygrad_cpu(
    instance: BoundInstance
) -> Tuple[Model, TokenizerWrapper, Callable[[Tensor], Tensor]]:

    model_id = instance.bound_shard.model_meta.model_id
    model_path = build_model_path(model_id)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # TODO: implement the contretized tokenizer wrapper class
    tokenizer = TokenizerWrapper(hf_tokenizer)

    # For a shared mode, we need to load all safetensors files.
    weights = {}
    for file in model_path.glob("*.safetensors"):
        weights.update(safe_load(str(file)))

    # TODO: implement and/or import the transformer class
    model = Transformer(...)

    load_state_dict(model, weights)

    def sampler(logits: Tensor) -> Tensor:
        return logits.argmax(axis=-1)

    return model, tokenizer, sampler
