"""Independently reload checkpoints and verify reported results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from corrected_experiments import ROOT, DATASETS, SEEDS, code_hash, digest, evaluate, load_corpus, select_prompts, save_json
from generation_quality_benchmark import evaluate_generation
from multi_dataset_fair_experiments import get_model_specs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=ROOT/'artifacts/corrected_v1')
    args=ap.parse_args()
    torch.set_num_threads(1)
    models={name:factory for name,_,factory in get_model_specs()}
    checked=[]
    for dataset in DATASETS:
        (_,val,test),vocab,metadata=load_corpus(dataset)
        starts,prompts,refs=select_prompts(test,64,160,32)
        decode=lambda row:''.join(vocab[int(t)] for t in row)
        for path in sorted((args.results/dataset).glob('*.json')):
            r=json.loads(path.read_text())
            assert r['code_sha256']==code_hash()
            assert r['corpus']==metadata
            cp=path.parent/r['checkpoint']
            assert digest(cp)==r['checkpoint_sha256']
            model=models[r['model']](len(vocab))
            model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
            test_loss,count=evaluate(model,test)
            val_loss,_=evaluate(model,val)
            assert abs(test_loss-r['final_test_loss'])<1e-7
            assert abs(val_loss-r['best_val_loss'])<1e-7
            assert count==len(test)-1==r['evaluation_tokens']
            assert r['best_epoch']==min(r['epoch_history'],key=lambda h:h['val_loss'])['epoch']
            for mode in ['greedy','sampled']:
                assert len(r['generation'][mode])==32
                for sample,start,prompt,ref in zip(r['generation'][mode],starts,prompts,refs):
                    assert sample['start_index']==start
                    assert sample['prompt']==decode(prompt)
                    assert sample['reference']==decode(ref)
                    assert len(sample['continuation'])==160
                    assert sample['metrics']==evaluate_generation(sample['continuation'],sample['reference'])
            checked.append((dataset,r['model'],r['seed']))
            print('Verified',dataset,r['model'],r['seed'],flush=True)
    expected={(d,m,s) for d in DATASETS for m in models for s in SEEDS}
    assert set(checked)==expected and len(checked)==72
    save_json(args.results/'integrity.json',dict(checkpoints=72,heldout_losses_recomputed=72,
        validation_losses_recomputed=72,continuations_checked=4608,
        code_sha256=code_hash(),all_passed=True))
    print('PASS: all 72 checkpoints, losses, prompts and 4608 diagnostic records')

if __name__=='__main__':
    main()
