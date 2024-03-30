import pytest
from train import get_all_sentences

def test_get_all_sentences(ds_raw_small, tgt_lang):
    generator = get_all_sentences(ds_raw_small, tgt_lang)
    sentences = list(generator)
    expected_output = ['Bonjour', 'Au revoir']
    assert sentences == expected_output
    
def test_get_all_sentences(ds_raw_small, src_lang):
    generator = get_all_sentences(ds_raw_small, src_lang)
    sentences = list(generator)
    expected_output = ['Hello', 'Goodbye']
    assert sentences == expected_output