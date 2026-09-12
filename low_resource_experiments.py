"""Compatibility entry point. Scientific execution uses corrected_experiments."""
from __future__ import annotations
import torch
from primitive_usage_helpers import build_sequence_dataset, sample_batch, next_token_cross_entropy

def get_device():
    # This correction series is explicitly a CPU experiment; no silent device fallback.
    return torch.device('cpu')
def count_parameters(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)
def generate_text_on_device(model,dataset,device,prompt,max_new_tokens,temperature=None):
    from corrected_experiments import generate
    if device.type!='cpu':raise ValueError('Corrected protocol currently requires CPU')
    ids=[dataset.stoi[c] for c in prompt]
    if not ids:raise ValueError('Prompt must not be empty')
    tokens=generate(model,torch.tensor([ids]),max_new_tokens,temperature)
    return prompt+''.join(dataset.itos[int(t)] for t in tokens[0])
if __name__=='__main__':
    from corrected_experiments import main
    main()
