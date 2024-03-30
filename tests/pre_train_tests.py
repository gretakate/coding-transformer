# What things do I want to test?
# - Correct output shapes
# - That there's no test dataset leakage
# - Bilingual dataset produces the correct data

import pytest
from train import get_all_sentences
   

@pytest.fixture
def sample_dataset():
    return [
        {'translation': {'en': 'Hello', 'fr': 'Bonjour'}},
        {'translation': {'en': 'Goodbye', 'fr': 'Au revoir'}}
    ]

def test_get_all_sentences(sample_dataset):
    generator = get_all_sentences(sample_dataset, 'fr')
    sentences = list(generator)
    expected_output = ['Bonjour', 'Au revoir']
    assert sentences == expected_output