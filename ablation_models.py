"""Prespecified intervention grid; the original model implementations stay unchanged."""
from __future__ import annotations
from itertools import product
import torch
from tiny_transformer import TinyTransformerDecoderLanguageModel
from multi_dataset_fair_experiments import get_model_specs

LEARNING_RATES = (0.0003, 0.001, 0.003)
DROPOUTS = (0.0, 0.1)
WIDTHS = (48, 32)
BASE_NAMES = {'transformer': 'Tiny Transformer Decoder Fair',
              'rnn': 'Primitive RNN Fair', 'gru': 'Primitive GRU Fair'}

def configurations():
    result = []
    for width, dropout, rate in product(WIDTHS, DROPOUTS, LEARNING_RATES):
        result.append(dict(architecture='transformer', width=width,
                           dropout=dropout, learning_rate=rate))
    for architecture, rate in product(('rnn', 'gru'), LEARNING_RATES):
        result.append(dict(architecture=architecture, learning_rate=rate))
    for item in result:
        item['id'] = (f"{item['architecture']}_w{item.get('width', 0)}_"
                      f"p{item.get('dropout', 0):.1f}_lr{item['learning_rate']:.4f}")
    return result

def transformer_count(vocabulary, width, feedforward):
    # Untied token/output weights, learned positions, one pre-norm block.
    return 2 * vocabulary * width + vocabulary + 64 * width + 4 * width**2 + 11 * width + (2 * width + 1) * feedforward

def dimensions(config, vocabulary):
    if config['architecture'] != 'transformer':
        return {}
    width = config['width']
    target = transformer_count(vocabulary, 48, 112)
    feedforward = min(range(1, 1025), key=lambda f: (abs(transformer_count(vocabulary, width, f)-target), f))
    return dict(d_model=width, ff_dim=feedforward, num_heads=4, num_layers=1,
                max_seq_len=64, dropout=config['dropout'])

def make_model(config, vocabulary):
    if config['architecture'] == 'transformer':
        return TinyTransformerDecoderLanguageModel(vocab_size=vocabulary, **dimensions(config, vocabulary))
    factory = next(f for name, _, f in get_model_specs() if name == BASE_NAMES[config['architecture']])
    return factory(vocabulary)

def is_original(config):
    return config['learning_rate'] == 0.001 and (config['architecture'] != 'transformer' or
           (config['width'] == 48 and config['dropout'] == 0.1))
