import pytest
from train import build_tokenizer
from model import build_transformer


@pytest.fixture
def ds_raw_small():
    # Dummy dataset for testing
    return [
        {'translation': {'en': 'Hello', 'fr': 'Bonjour'}},
        {'translation': {'en': 'Goodbye', 'fr': 'Au revoir'}}
    ]
    
@pytest.fixture
def ds_raw():
    # Expanded dummy dataset for testing (20 samples)
    return [
        {'translation': {'en': 'Hello', 'fr': 'Bonjour'}},
        {'translation': {'en': 'Goodbye', 'fr': 'Au revoir'}},
        {'translation': {'en': 'How are you?', 'fr': 'Comment ça va?'}},
        {'translation': {'en': 'I am fine, thank you.', 'fr': 'Je vais bien, merci.'}},
        {'translation': {'en': 'What is your name?', 'fr': 'Comment vous appelez-vous?'}},
        {'translation': {'en': 'My name is John.', 'fr': 'Je m\'appelle Jean.'}},
        {'translation': {'en': 'Where are you from?', 'fr': 'D\'où venez-vous?'}},
        {'translation': {'en': 'I am from Paris.', 'fr': 'Je viens de Paris.'}},
        {'translation': {'en': 'Can you help me, please?', 'fr': 'Pouvez-vous m\'aider, s\'il vous plaît?'}},
        {'translation': {'en': 'Sure, what do you need?', 'fr': 'Bien sûr, de quoi avez-vous besoin?'}},
        {'translation': {'en': 'Thank you!', 'fr': 'Merci!'}},
        {'translation': {'en': 'You\'re welcome.', 'fr': 'De rien.'}},
        {'translation': {'en': 'Have a nice day!', 'fr': 'Bonne journée!'}},
        {'translation': {'en': 'Good night.', 'fr': 'Bonne nuit.'}},
        {'translation': {'en': 'See you later!', 'fr': 'À plus tard!'}},
        {'translation': {'en': 'Where is the nearest restaurant?', 'fr': 'Où est le restaurant le plus proche?'}},
        {'translation': {'en': 'It\'s just around the corner.', 'fr': 'C\'est juste au coin.'}},
        {'translation': {'en': 'I\'m lost.', 'fr': 'Je suis perdu.'}},
        {'translation': {'en': 'Can you show me the way to the train station?', 'fr': 'Pouvez-vous me montrer le chemin de la gare?'}},
        {'translation': {'en': 'Excuse me, where is the bathroom?', 'fr': 'Excusez-moi, où sont les toilettes?'}}                
    ]
    
@pytest.fixture
def src_lang():
    return 'en'

@pytest.fixture
def tgt_lang():
    return 'fr'

@pytest.fixture
def device():
    return 'cpu'  # Dummy device

@pytest.fixture
def config(src_lang, tgt_lang):
    return {
        'batch_size': 5,
        "d_ff": 2048,
        "d_model": 512,
        'datasource': 'dataset.txt',
        "dropout": 0.1,
        'experiment_name': 'test_experiment',
        "h": 8,
        'lang_src': src_lang,
        'lang_tgt': tgt_lang,
        "lr": 10**-4,
        "model_basename": "tmodel_test_",
        "model_folder": "weights_test",
        "N": 6,
        "num_epochs": 2,
        'preload': None,
        'seq_len': 50,
        'tokenizer_file': 'tokenizer_{}.json',
    }

@pytest.fixture
def tokenizer_src(ds_raw, src_lang):
    """Creates a fixture for the source language tokenizer."""
    return build_tokenizer(ds_raw, src_lang)

@pytest.fixture
def tokenizer_tgt(ds_raw, tgt_lang):
    """Creates a fixture for the source language tokenizer."""
    return build_tokenizer(ds_raw, tgt_lang)

@pytest.fixture
def dummmy_model():
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    return build_transformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'], config['N'], config['h'], config['dropout'], config['d_ff'])
