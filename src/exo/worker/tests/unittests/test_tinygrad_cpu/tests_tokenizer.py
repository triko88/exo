import unittest
import sys
from unittest.mock import MagicMock

# Mock tinygrad before importing modules that depend on it
tinygrad_mock = MagicMock()
sys.modules["tinygrad"] = tinygrad_mock
sys.modules["tinygrad.tensor"] = MagicMock()
sys.modules["tinygrad.device"] = MagicMock()

from exo.worker.engines.tinygrad_cpu.utils_tokenizer import TinygradTokenizer, TinygradDetokenizer

class TestTinygradTokenizer(unittest.TestCase):
    def setUp(self):
        # Mock the HF tokenizer
        self.mock_hf = MagicMock()
        self.mock_hf.bos_token = "<s>"
        self.mock_hf.eos_token = 2  # Matches usage in utils_tokenizer.py: self.eos_token_ids.append(hf_tokenizer.eos_token)
        
        # Setup common return values
        self.mock_hf.encode.return_value = [1, 2, 3]
        self.mock_hf.apply_chat_template.return_value = "Formatted Prompt"
        self.mock_hf.apply_message_template.return_value = "Formatted Prompt" # Only if the wrapper delegates to a method with this name, but standard HF uses apply_chat_template. 
        # CAUTION: The user's code calls `self.tokenizer.apply_message_template`, which is NON-STANDARD for HF tokenizers. 
        # Verify if the user intended to call `apply_chat_template` or if they really have a tokenizer with `apply_message_template`.
        # Assuming standard HF, the user's code is likely buggy here:
        # return self.tokenizer.apply_message_template(...) -> likely should be apply_chat_template
        # But specifically testing the WRAPPER code logic:
        self.mock_hf.apply_message_template.return_value = "Mocked Template Response"
        
        self.tokenizer = TinygradTokenizer(self.mock_hf)

    def test_initialization(self):
        self.assertEqual(self.tokenizer.bos_token, "<s>")
        self.assertEqual(self.tokenizer.eos_token_ids, [2])
        self.assertIsInstance(self.tokenizer.detokenizer, TinygradDetokenizer)

    def test_encode(self):
        result = self.tokenizer.encode("test")
        self.mock_hf.encode.assert_called_with("test", add_special_tokens=True)
        self.assertEqual(result, [1, 2, 3])

    def test_apply_message_template(self):
        messages = [{"role": "user", "content": "hi"}]
        result = self.tokenizer.apply_message_template(messages)
        # Usage in code: return self.tokenizer.apply_message_template(...)
        self.mock_hf.apply_message_template.assert_called_with(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=False
        )
        self.assertEqual(result, "Mocked Template Response")

class TestTinygradDetokenizer(unittest.TestCase):
    def setUp(self):
        self.mock_hf = MagicMock()
        self.detokenizer = TinygradDetokenizer(self.mock_hf)
    
    def test_accumulate_tokens(self):
        # Simulation: "H" -> "He" -> "Hel" -> "Hell" -> "Hello"
        # tokenizer.decode called with [1], then [1,2], etc.
        
        # Step 1: Add token 1 -> "H"
        self.mock_hf.decode.return_value = "H"
        self.detokenizer.add(1)
        self.assertEqual(self.detokenizer.get(), "H")
        self.assertEqual(self.detokenizer.last_segment, "H")
        
        # Step 2: Add token 2 -> "He"
        self.mock_hf.decode.return_value = "He"
        self.detokenizer.add(2)
        self.assertEqual(self.detokenizer.get(), "He")
        self.assertEqual(self.detokenizer.last_segment, "e")

    def test_reset(self):
        self.detokenizer.add(1)
        self.detokenizer.reset()
        self.assertEqual(self.detokenizer.get(), "")
        self.assertEqual(self.detokenizer.all_tokens, [])

if __name__ == "__main__":
    unittest.main()
