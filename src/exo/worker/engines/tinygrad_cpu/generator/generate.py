from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.runner_response import GenerationResponse
from exo.worker.engines.tinygrad_cpu import Model
from exo.worker.engines.tinygrad_cpu.constants import MAX_TOKENS
from exo.worker.engines.tinygrad_cpu.utils_tokenizer import TinygradTokenizer
from tinygrad.tensor import Tensor
from typing import Callable, Generator


def warmup_inference(
    model: Model,
    tokenizer: TinygradTokenizer,
    sampler: Callable[[Tensor], Tensor]) -> int:

    task = ChatCompletionTaskParams(
        model="",
        messages=[
            ChatCompletionMessage(
                role="user",
                content="Prompt to warm up the inference engine. Repeat this.",
            )
        ],
        max_tokens=10,
    )

    count = 0
    for _ in tinygrad_generate(model, tokenizer, sampler, task):
        count += 1
    return count

def tinygrad_generate(
    model: Model,
    tokenizer: TinygradTokenizer,
    sampler: Callable[[Tensor], Tensor],
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse]:

    prompt_text = tokenizer.apply_message_template(
        task.messages,
        add_generation_prompt=True,
    )
    tokens = tokenizer.encode(prompt_text)

    tokenizer.detokenizer.reset()
    
    start_pos = 0
    max_tokens = task.max_tokens or MAX_TOKENS
    curr_tokens = tokens[:]

    for idx in ranges(max_tokens):
        if idx == 0:
            x = Tensor([curr_tokens]).realize()
            # Gets shape as (1, seq_len)
        else:
            x = Tensor([curr_tokens[-1]]).realize()
            # Gets shape as (1, 1)

        logits = model(x, start_pos=start_pos)

        # Sample
        next_token_tensor = sampler(logits[:, -1, :])
        next_token = int(next_token_tensor.item())

        # Update state
        curr_tokens.append(next_token)
        start_pos += x.shape[1]

        # Detokenize
        tokenizer.detokenizer.add(next_token)
        new_text = tokenizer.detokenizer.last_segment

        # Check EOS
        finish_reason = None
        if tokenizer.eos_token_ids and next_token in tokenizer.eos_token_ids:
            finish_reason = "stop"
        elif idx == max_tokens - 1:
            finish_reason = "length"

        
        yield GenerationResponse(
            text=new_text,
            token=next_token,
            finish_reason=finish_reason,
        )

        if finish_reason:
            break
