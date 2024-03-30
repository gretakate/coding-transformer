import pytest
from train import get_ds, build_tokenizer
from unittest.mock import patch

    
def get_or_build_tokenizer_mock_function(config, ds_raw, src_lang):
    return build_tokenizer(ds_raw, src_lang)
    

def test_get_ds(config, ds_raw):
    
    with patch('train.get_or_build_tokenizer') as get_or_build_tokenizer_mock:
        get_or_build_tokenizer_mock.side_effect = get_or_build_tokenizer_mock_function
        
        with patch('train.load_dataset') as load_dataset_mock:
            load_dataset_mock.return_value = ds_raw
            
            # Call the function under test
            train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
            
            # Assert that get_or_build_tokenizer was called twice with correct arguments
            get_or_build_tokenizer_mock.assert_any_call(config, ds_raw, config['lang_src'])
            get_or_build_tokenizer_mock.assert_any_call(config, ds_raw, config['lang_tgt'])
            
            # Assert that the returned objects are not None
            assert train_dataloader is not None
            assert val_dataloader is not None
            assert tokenizer_src is not None
            assert tokenizer_tgt is not None
            
            # Assert that the batch sizes are correct
            assert train_dataloader.batch_size == config['batch_size']
            assert val_dataloader.batch_size == 1
            
            # Assert that the train_dataloader has the correct keys
            total_training_samples = 0
            for batch in train_dataloader:
                total_training_samples += len(batch['encoder_input'])
                assert 'encoder_input' in batch
                assert 'decoder_input' in batch
                assert 'encoder_mask' in batch
                assert 'decoder_mask' in batch
                assert 'src_text' in batch
                assert 'tgt_text' in batch
                assert 'label' in batch
                
            # Assert that the val_dataloader has the correct keys
            total_validation_samples = 0
            for batch in val_dataloader:
                total_validation_samples += len(batch['encoder_input'])
                assert 'encoder_input' in batch
                assert 'decoder_input' in batch
                assert 'encoder_mask' in batch
                assert 'decoder_mask' in batch
                assert 'src_text' in batch
                assert 'tgt_text' in batch
                assert 'label' in batch
                
            # Assert that the correct 90/10 training/validation split happened
            assert total_training_samples == int(0.9 * len(ds_raw))
            assert total_validation_samples == len(ds_raw) - total_training_samples
            
            # Make sure there is no training/validation leakage
            training_encoder_input_set = set()
            for batch in train_dataloader:
                for txt in batch['src_text']:
                    training_encoder_input_set.add(txt)
            
            val_encoder_input_set = set()
            for batch in val_dataloader:
                for txt in batch['src_text']:
                    val_encoder_input_set.add(txt)
            
            assert len(training_encoder_input_set & val_encoder_input_set) == 0

            
            