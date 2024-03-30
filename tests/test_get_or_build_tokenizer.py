import os
import tempfile

from train import get_or_build_tokenizer

def test_get_or_build_tokenizer_exists(config, ds_raw, src_lang, tokenizer_src):
    # Create a temporary directory to store the tokenizer file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update the config to point to this temporary directory
        tokenizer_file = os.path.join(temp_dir, config['tokenizer_file'].format(src_lang))
        
        # Create an empty tokenizer file
        tokenizer_src.save(str(tokenizer_file))
        
        # Make sure that the path exists prior to calling get_or_build_tokenizer
        assert os.path.exists(tokenizer_file)
        
        config['tokenizer_file'] = tokenizer_file
        tokenizer = get_or_build_tokenizer(config, ds_raw, src_lang)
        
        # Assert that the existing tokenizer file was loaded
        assert os.path.exists(tokenizer_file)
        assert tokenizer is not None
        
def test_get_or_build_tokenizer_not_exists(config, ds_raw, src_lang):
    # Create a temporary directory to store the tokenizer file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update the config to point to this temporary directory
        config['tokenizer_file'] = os.path.join(temp_dir, config['tokenizer_file'].format(src_lang))
        tokenizer_file = config['tokenizer_file']
        
        # Make sure that the path does not prior to calling get_or_build_tokenizer
        assert not os.path.exists(tokenizer_file)
        
        tokenizer = get_or_build_tokenizer(config, ds_raw, src_lang)
        
        # Assert that a new tokenizer file was created
        assert os.path.exists(tokenizer_file)
        assert tokenizer is not None
        