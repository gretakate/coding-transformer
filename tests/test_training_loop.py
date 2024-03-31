from train import train_model
from unittest.mock import patch
import tempfile
import os


def test_train_model(config, ds_raw):
    # Create a temporary directory to store the tokenizer file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set config to map to this temp_dir so files are cleaned up after testing
        config['experiment_name'] = os.path.join(temp_dir, config['experiment_name'])
        config['model_folder'] = os.path.join(temp_dir, config['model_folder'])
        config['tokenizer_file'] = os.path.join(temp_dir, config['tokenizer_file'])
        
        # Force the test to choose CPU
        with patch('train.get_device', return_value='cpu'):
            # Load the mock dataset instead of reaching out to HuggingFace
            with patch('train.load_dataset') as load_dataset_mock:
                load_dataset_mock.return_value = ds_raw
                train_model(config)
                
                # Assert that the tokenizer files were created
                assert os.path.exists(config['tokenizer_file'].format(config['lang_src']))
                assert os.path.exists(config['tokenizer_file'].format(config['lang_tgt']))
                
                # Assert that the weights files were saved
                assert os.path.exists(config['model_folder'])
        
    # Make sure that the training loop finishes without exceptions
    assert True