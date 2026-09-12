"""Verify selection chronology, paired controls and all ablation checkpoints."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
from pathlib import Path
import torch
import ablation_experiments as exp
from ablation_models import make_model, is_original, dimensions, BASE_NAMES
import corrected_experiments as base
from generation_quality_benchmark import evaluate_generation


def verify_job(args):
    output,path,record,selection=args
    torch.set_num_threads(1)
    d,c,s=record['dataset'],record['config'],record['seed']
    checkpoint=path.parent/record['checkpoint']
    result=json.loads((output/'evaluation'/d/(exp.key(c,s)+'.json')).read_text())
    assert result['training_sha256']==base.digest(path)
    assert result['selection_sha256']==base.digest(output/'selection.json')
    assert result['evaluated_at']>=selection['locked_at']
    assert record['completed_at']<=selection['locked_at']
    assert not any(k in record for k in ['test_loss','final_test_loss','generation'])
    streams,vocab,meta=base.load_corpus(d)
    assert record['corpus']==meta
    torch.manual_seed(s);model=make_model(c,len(vocab))
    assert exp.state_hash(model)==record['initial_state_sha256']
    assert sum(p.numel() for p in model.parameters())==record['params']==result['params']
    assert dimensions(c,len(vocab))==record['dimensions']
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    for stream,name in zip(streams,['train_loss','val_loss','test_loss']):
        value,count=base.evaluate(model,stream)
        assert abs(value-result[name])<1e-7,(d,c['id'],s,name)
        if name=='test_loss':assert count==result['test_targets']==len(stream)-1
        if name=='train_loss':assert count==result['train_targets']==len(stream)-1
    if is_original(c):
        original_name=BASE_NAMES[c['architecture']].lower().replace(' ','_')+f'_seed_{s}.json'
        old=json.loads((exp.BASELINE/d/original_name).read_text())
        assert base.digest(checkpoint)==old['checkpoint_sha256']
        assert abs(result['test_loss']-old['final_test_loss'])<1e-7
    generated=0
    if c['id']==selection['selected'][c['architecture']]:
        starts,prompts,refs=base.select_prompts(streams[2],64,160,32)
        decode=lambda row:''.join(vocab[int(t)] for t in row)
        for mode,temp in [('greedy',None),('sampled',.8)]:
            sequences=base.generate(model,prompts,160,temp,s+200000,64)
            samples=result['generation'][mode];assert len(samples)==32
            for i,sample in enumerate(samples):
                assert sample['start_index']==starts[i]
                assert sample['prompt']==decode(prompts[i])
                assert sample['reference']==decode(refs[i])
                assert sample['continuation']==decode(sequences[i])
                assert sample['metrics']==evaluate_generation(sample['continuation'],sample['reference'])
                generated+=1
    else:assert result['generation']=={}
    return generated


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=exp.OUTPUT);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();output=args.results.resolve()
    frozen=json.loads((output/'plan.json').read_text())
    assert frozen['source_hash']==exp.source_hash()
    records=exp.collect(output,frozen);assert len(records)==216
    selection=json.loads((output/'selection.json').read_text())
    assert exp.lock_selection(output,frozen)==selection
    assert sum(r['reused_original'] for _,r in records)==36
    groups=defaultdict(set)
    for p,r in records:
        assert selection['training_records'][str(p.relative_to(output))]==base.digest(p)
        c=r['config'];groups[r['dataset'],c['architecture'],c.get('width'),r['seed']].add(r['initial_state_sha256'])
    assert all(len(g)==1 for g in groups.values())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        counts=list(pool.map(verify_job,[(output,p,r,selection) for p,r in records]))
    assert sum(counts)==2304
    value=dict(all_passed=True,training_records=216,new_training_runs=180,reused_runs=36,
               checkpoints_reloaded=216,train_validation_test_losses_recomputed=648,
               selected_continuations_regenerated=2304,selection_before_test=True,
               matched_initialization_groups=len(groups),source_hash=frozen['source_hash'],
               selection_sha256=base.digest(output/'selection.json'))
    base.save_json(output/'integrity.json',value);print(json.dumps(value,indent=2))
if __name__=='__main__':main()
