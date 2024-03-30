# What things do I want to test?
# - Correct output shapes
# - That there's no test dataset leakage
# - Bilingual dataset produces the correct data

import os
import shutil
import tempfile
import pytest
from train import get_all_sentences, get_or_build_tokenizer
   
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