# What things do I want to test?
# - Correct output shapes
# - That there's no test dataset leakage
# - Bilingual dataset produces the correct data

import os
import shutil
import tempfile
import pytest
import torch

from train import get_all_sentences, get_or_build_tokenizer, greedy_decode
   
@pytest.fixture
def config():
    return {
        'tokenizer_file': 'tokenizer_{}.json'  # Example tokenizer file path format
    }
    
@pytest.fixture
def lang():
    return 'fr'
    
@pytest.fixture
def dataset():
    return [
        {'translation': {'en': 'Hello', 'fr': 'Bonjour'}},
        {'translation': {'en': 'Goodbye', 'fr': 'Au revoir'}}
    ]
    
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

@pytest.fixture
def device():
    return 'cpu'  # Dummy device

def test_get_all_sentences(dataset):
    generator = get_all_sentences(dataset, 'fr')
    sentences = list(generator)
    expected_output = ['Bonjour', 'Au revoir']
    assert sentences == expected_output
    
def test_get_or_build_tokenizer_exists(config, dataset, lang):
    # Create a temporary directory to store the tokenizer file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update the config to point to this temporary directory
        tokenizer_file = os.path.join(temp_dir, config['tokenizer_file'].format(lang))
        
        # Create an empty tokenizer file
        open(tokenizer_file, 'a').close()
        
        # Make sure that the path exists prior to calling get_or_build_tokenizer
        assert os.path.exists(tokenizer_file)
        
        tokenizer = get_or_build_tokenizer(config, dataset, lang)
        
        # Assert that the existing tokenizer file was loaded
        assert os.path.exists(tokenizer_file)
        assert tokenizer is not None
        
def test_get_or_build_tokenizer_not_exists(config, dataset, lang):
    # Create a temporary directory to store the tokenizer file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update the config to point to this temporary directory
        config['tokenizer_file'] = os.path.join(temp_dir, config['tokenizer_file'].format(lang))
        tokenizer_file = config['tokenizer_file']
        
        # Make sure that the path does not prior to calling get_or_build_tokenizer
        assert not os.path.exists(tokenizer_file)
        
        tokenizer = get_or_build_tokenizer(config, dataset, lang)
        
        # Assert that a new tokenizer file was created
        assert os.path.exists(tokenizer_file)
        assert tokenizer is not None
        


def test_greedy_decode_gives_output_correct_max_len(model, source, source_mask, tokenizer_tgt, max_len, device):
    # Call the function under test
    decoded_sequence = greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device)

    # Assert that the decoded sequence has the correct shape
    assert decoded_sequence.shape == (max_len,)  # We expect the decoded sequence to have a shape of (max_len,)
