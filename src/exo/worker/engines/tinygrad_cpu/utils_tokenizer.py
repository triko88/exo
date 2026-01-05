from typing import Any
from exo.worker.engines.tinygrad_cpu import Detokenizer, TokenizerWrapper

class TinygradTokenizer(TokenizerWrapper):
    def __init__(self, hf_tokenizer: Any):
        self.tokenizer = hf_tokenizer
        self.detokenizer = TinygradDetokenizer(self.tokenizer)
        self.bos_token = hf_tokenizer.bos_token
        self.eos_token_ids = []

        if hf_tokenizer.eos_token is not None:
            self.eos_token_ids.append(hf_tokenizer.eos_token)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def apply_message_template(
        self,
        message_dicts: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        return self.tokenizer.apply_message_template(
            conversation=message_dicts, 
            tokenize=tokenize, 
            add_generation_prompt=add_generation_prompt)

class TinygradDetokenizer(Detokenizer):
    all_tokens: list[int]
    full_text: str
    current_segment: str

    def __init__(self, tokenizer: Any):
        self.all_tokens = []
        self.full_text = ""
        self.current_segment = ""
        self.tokenizer = tokenizer

    def reset(self) -> None: 
        self.all_tokens = []
        self.full_text = ""

    def add(self, token: int) -> None:
        self.all_tokens.append(token)
        new_text = self.tokenizer.decode(self.all_tokens)

        if len(new_text) > len(self.full_text):
            self.current_segment = new_text[len(self.full_text):]
            self.full_text = new_text
        else:
            self.current_segment = ""

    def get(self) -> str:
        return self.full_text
    
    @property
    def last_segment(self) -> str:
        return self.current_segment
