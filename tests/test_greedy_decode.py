import pytest
import torch

from train import greedy_decode
   
@pytest.fixture
def model():
    # Define a dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()

        def encode(self, source, source_mask):
            return torch.zeros((1, source.size(1), 512))  # Dummy encoder output

        def decode(self, encoder_output, source_mask, decoder_input, decoder_mask):
            return torch.zeros((1, decoder_input.size(1) + 1, 512))  # Dummy decoder output

        def project(self, decoder_output):
            return torch.ones((1, 1000))  # Dummy project output

    return DummyModel()

@pytest.fixture
def source():
    return torch.tensor([[1, 2, 3]])  # Dummy source sequence tensor

@pytest.fixture
def source_mask():
    return torch.ones((1, 1, 3))  # Dummy source mask tensor

@pytest.fixture
def tokenizer_tgt():
    # Define a dummy target tokenizer for testing
    class DummyTokenizer:
        def token_to_id(self, token):
            return 1  # Dummy token ID

    return DummyTokenizer()

@pytest.fixture
def max_len():
    return 10  # Maximum length of the generated sequence


def test_greedy_decode_gives_output_correct_max_len(model, source, source_mask, tokenizer_tgt, max_len, device):
    # Call the function under test
    decoded_sequence = greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device)

    # Assert that the decoded sequence has the correct shape
    assert decoded_sequence.shape == (max_len,)  # We expect the decoded sequence to have a shape of (max_len,)
