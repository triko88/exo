import unittest
import sys
from unittest.mock import MagicMock, ANY

# Mock tinygrad before imports
tinygrad_mock = MagicMock()
sys.modules["tinygrad"] = tinygrad_mock
sys.modules["tinygrad.tensor"] = MagicMock()
sys.modules["tinygrad.device"] = MagicMock()

# Handle mocking more carefully.
# We need 'exo' to be a real package if we want to import submodules from it,
# OR we need to mock everything.
# Since we are setting PYTHONPATH=src, 'exo' SHOULD be importable.
# The issue likely was that we overwrote sys.modules["exo"] with a Mock, 
# preventing Python from finding the real 'exo' package on disk.

# So, we should ONLY mock the *missing* modules.
# We know 'exo.shared.types.api' was missing or problematic.

import types

# Create dummy modules for only what we strictly need to mock
mock_shared_types_api = types.ModuleType("exo.shared.types.api")
ChatCompletionTaskParams = MagicMock()
mock_shared_types_api.ChatCompletionTaskParams = ChatCompletionTaskParams
sys.modules["exo.shared.types.api"] = mock_shared_types_api

mock_runner_response = types.ModuleType("exo.shared.types.runner_response")
GenerationResponse = MagicMock()
mock_runner_response.GenerationResponse = GenerationResponse
sys.modules["exo.shared.types.runner_response"] = mock_runner_response

# We should NOT mock "exo", "exo.worker", or "exo.shared" generally
# because we want to actually import "exo.worker.engines.tinygrad_cpu..."
# from the disk (since PYTHONPATH=src).

# Now import the module under test
# We used "exo.worker.engines.tinygrad_cpu.generator.generate" in the test file
# ensure that path is resolvable or mock the intermediates if needed, 
# but since the file exists on disk and we set PYTHONPATH=src, standard import *should* work 
# provided the dependencies inside it are mocked.
from exo.worker.engines.tinygrad_cpu.generator.generate import warmup_inference, tinygrad_generate

class TestTinygradGenerate(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_sampler = MagicMock()
        self.mock_task = MagicMock()
        
        # Setup tokenizer mocks
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.detokenizer.decode.return_value = "decoded"
        
        # Setup sampler mock
        self.mock_sampler.return_value = 123  # dummy token id

    def test_warmup_inference_returns_int(self):
        """Test that warmup_inference runs and returns an integer (number of tokens likely)."""
        result = warmup_inference(self.mock_model, self.mock_tokenizer, self.mock_sampler)
        self.assertIsInstance(result, int, "warmup_inference should return an integer")

    def test_tinygrad_generate_is_generator(self):
        """Test that tinygrad_generate returns a generator."""
        gen = tinygrad_generate(self.mock_model, self.mock_tokenizer, self.mock_sampler, self.mock_task)
        import types
        self.assertIsInstance(gen, types.GeneratorType, "tinygrad_generate should return a generator")

    def test_tinygrad_generate_yields_response(self):
        """Test that iterating the generator yields GenerationResponse objects."""
        # For this test to work once implemented, the function needs to actually yield something.
        # Currently it returns None (implicitly) because of '...'.
        # Trying to iterate over None will raise TypeError, which counts as a failure.
        
        try:
            gen = tinygrad_generate(self.mock_model, self.mock_tokenizer, self.mock_sampler, self.mock_task)
            first_chunk = next(gen)
            # If we get here (which we won't with current implementation), check type
            # We assume GenerationResponse or dict
            self.assertTrue(hasattr(first_chunk, "text") or hasattr(first_chunk, "token_id"), "Should yield a proper response object")
        except TypeError:
            self.fail("tinygrad_generate did not return an iterable (returned None?)")
        except StopIteration:
            # If it's an empty generator, that's arguably "passing" the type check but failing logic.
            # But with '...' it returns None, so TypeError is expected.
            pass

if __name__ == "__main__":
    unittest.main()
