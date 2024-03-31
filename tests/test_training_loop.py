from train import train_model
from unittest.mock import patch


def test_train_model(config, ds_raw):
    # Force the test to choose CPU
    with patch('train.get_device', return_value='cpu'):
        # Load the mock dataset
        with patch('train.load_dataset') as load_dataset_mock:
            load_dataset_mock.return_value = ds_raw
            train_model(config)
        
    assert True