"""Train a frozen ablation grid, lock validation choices, then evaluate test data.

Existing original runs are reused only after verifying data/code/checkpoint hashes.
Training records are immutable inputs to selection; test scores live separately.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
from torch.nn import functional as F
import corrected_experiments as base
from ablation_models import BASE_NAMES, configurations, dimensions, is_original, make_model

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'artifacts/ablation_v1'
BASELINE = ROOT/'artifacts/corrected_v1'
CONFIGS = configurations()

def source_hash():
    text = base.code_hash()
    for name in ['ablation_models.py', 'ablation_experiments.py']:
        text += name + base.digest(ROOT/name)
    return hashlib.sha256(text.encode()).hexdigest()

def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def key(config, seed):
    return config['id'] + f'_seed_{seed}'

def plan():
    return dict(version=1,configs=CONFIGS,seeds=base.SEEDS,datasets=base.DATASETS,
        training_protocol=base.CONFIG,source_hash=source_hash(),baseline_source_hash=base.code_hash(),
        design='Transformer 2 dropout x 3 learning rates x 2 allocations; RNN/GRU same 3 learning rates',
        selection='One configuration per architecture, minimizing validation NLL averaged equally over corpora and seeds; ties by config id',
        primary_contrasts='Paired NLL differences for dropout, learning rate and allocation, marginalized over the other prespecified factors',
        baseline_disclosure='Follow-up designed after original baseline results were known; grid fixed before new ablation outcomes',
        interpretation='Effects of interventions under this training/selection policy, not universal architecture causality',
        generation='Greedy and temperature 0.8, 32 prompts, 64 context, 160 continuation, for validation-selected configurations only',
        corpus_hashes={d:base.load_corpus(d)[2]['sha256'] for d in base.DATASETS})

def record_paths(output, dataset, config, seed):
    directory=Path(output)/'training'/dataset
    return directory/(key(config,seed)+'.json'), directory/(key(config,seed)+'.pt')

def train_one(dataset, config, seed, output, frozen):
    torch.set_num_threads(1)
    (train,val,_),vocab,corpus=base.load_corpus(dataset)
    if corpus['sha256'] != frozen['corpus_hashes'][dataset]:raise ValueError('Corpus changed')
    path,checkpoint=record_paths(output,dataset,config,seed)
    path.parent.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(dict(plan=frozen,dataset=dataset,config=config,seed=seed),sort_keys=True).encode()).hexdigest()
    if path.exists():
        result=json.loads(path.read_text())
        if result['signature'] != signature:raise ValueError('Incompatible training record')
        if checkpoint.exists() and base.digest(checkpoint)==result['checkpoint_sha256']:return result
    torch.manual_seed(seed)
    model=make_model(config,len(vocab));initial_hash=state_hash(model)
    history=[]; reused=False;original=None
    started=time.perf_counter()
    if is_original(config):
        name=BASE_NAMES[config['architecture']].lower().replace(' ','_')+f'_seed_{seed}'
        old_path=BASELINE/dataset/(name+'.json')
        original=json.loads(old_path.read_text())
        old_cp=BASELINE/dataset/original['checkpoint']
        if original['code_sha256']!=frozen['baseline_source_hash'] or original['corpus']!=corpus:
            raise ValueError('Baseline code or corpus mismatch')
        if base.digest(old_cp)!=original['checkpoint_sha256']:raise ValueError('Baseline checkpoint mismatch')
        model.load_state_dict(torch.load(old_cp,map_location='cpu',weights_only=True))
        best,count=base.evaluate(model,val)
        if abs(best-original['best_val_loss'])>1e-7:raise ValueError('Baseline validation mismatch')
        history=original['epoch_history'];best_epoch=original['best_epoch'];stop=original['stop_reason']
        checkpoint.write_bytes(old_cp.read_bytes());reused=True
    else:
        cfg=frozen['training_protocol']
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        rng=torch.Generator().manual_seed(seed+100000)
        best=float('inf');patience_best=float('inf');stale=0;stop='max_epochs_reached'
        for epoch in range(1,cfg['max_epochs']+1):
            model.train();total=0.
            for step in range(cfg['steps_per_epoch']):
                x,y=base.batch(train,cfg['sequence_length'],cfg['batch_size'],rng)
                optimizer.zero_grad(set_to_none=True);z=model(x)
                loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1))
                if not bool(torch.isfinite(loss)):raise ValueError('Non-finite training loss')
                loss.backward();optimizer.step();total+=loss.item()
            value,count=base.evaluate(model,val,cfg['sequence_length'])
            best,patience_best,stale,save=base.checkpoint_decision(value,best,patience_best,stale,cfg)
            if save:
                temp=checkpoint.with_suffix('.pt.tmp');torch.save(model.state_dict(),temp);temp.replace(checkpoint);best_epoch=epoch
            history.append(dict(epoch=epoch,train_loss=total/cfg['steps_per_epoch'],val_loss=value))
            if stale>=cfg['patience']:stop='validation_plateau';break
    result=dict(signature=signature,source_hash=frozen['source_hash'],config=config,dataset=dataset,seed=seed,
        corpus=corpus,params=sum(p.numel() for p in model.parameters()),dimensions=dimensions(config,len(vocab)),
        initial_state_sha256=initial_hash,best_val_loss=best,best_epoch=best_epoch,stop_reason=stop,
        epoch_history=history,checkpoint=checkpoint.name,checkpoint_sha256=base.digest(checkpoint),
        reused_original=reused,original_signature=original['signature'] if reused else None,
        elapsed_seconds=time.perf_counter()-started,completed=True,
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=platform.python_version(),torch=torch.__version__,device='cpu',threads=1)
    base.save_json(path,result)
    print('TRAINED',dataset,config['id'],seed,'val',round(best,4),'reused' if reused else 'new',flush=True)
    return result

def collect(output,frozen):
    records=[]
    for d in frozen['datasets']:
        for c in frozen['configs']:
            for s in frozen['seeds']:
                path,cp=record_paths(output,d,c,s)
                r=json.loads(path.read_text())
                if not r['completed'] or r['source_hash']!=frozen['source_hash'] or base.digest(cp)!=r['checkpoint_sha256']:
                    raise ValueError('Incomplete or incompatible training results')
                if r['best_val_loss']!=min(h['val_loss'] for h in r['epoch_history']):raise ValueError('Not true minimum')
                records.append((path,r))
    return records

def lock_selection(output,frozen):
    records=collect(output,frozen)
    scores={c['id']:sum(r['best_val_loss'] for _,r in records if r['config']['id']==c['id'])/(len(base.DATASETS)*len(base.SEEDS)) for c in frozen['configs']}
    selected={a:min([c['id'] for c in frozen['configs'] if c['architecture']==a],key=lambda x:(scores[x],x)) for a in BASE_NAMES}
    value=dict(selected=selected,validation_means=scores,source_hash=frozen['source_hash'],
               training_records={str(p.relative_to(output)):base.digest(p) for p,_ in records})
    path=Path(output)/'selection.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=v for k,v in value.items()):raise ValueError('Locked selection changed')
        return old
    value['locked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    base.save_json(path,value);return value

def evaluate_one(dataset,config,seed,output,frozen,selected):
    torch.set_num_threads(1)
    path,checkpoint=record_paths(output,dataset,config,seed);r=json.loads(path.read_text())
    target=Path(output)/'evaluation'/dataset/(key(config,seed)+'.json')
    if target.exists():
        old=json.loads(target.read_text())
        if old['training_sha256']==base.digest(path) and old['selection_sha256']==base.digest(Path(output)/'selection.json'):return old
        raise ValueError('Stale evaluation')
    (train,val,test),vocab,meta=base.load_corpus(dataset)
    if meta!=r['corpus'] or base.digest(checkpoint)!=r['checkpoint_sha256']:raise ValueError('Source mismatch')
    model=make_model(config,len(vocab));model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    val_loss,_=base.evaluate(model,val)
    if abs(val_loss-r['best_val_loss'])>1e-7:raise ValueError('Checkpoint validation mismatch')
    train_loss,train_n=base.evaluate(model,train);test_loss,test_n=base.evaluate(model,test)
    generation={}
    if config['id']==selected[config['architecture']]:
        starts,prompts,refs=base.select_prompts(test,64,160,32)
        decode=lambda row:''.join(vocab[int(t)] for t in row)
        from generation_quality_benchmark import evaluate_generation
        for mode,temp in [('greedy',None),('sampled',.8)]:
            seqs=base.generate(model,prompts,160,temp,seed+200000,64)
            generation[mode]=[dict(start_index=s,prompt=decode(p),reference=decode(ref),continuation=decode(seq),
                metrics=evaluate_generation(decode(seq),decode(ref))) for s,p,ref,seq in zip(starts,prompts,refs,seqs)]
    result=dict(dataset=dataset,config=config,seed=seed,params=r['params'],train_loss=train_loss,val_loss=val_loss,
        test_loss=test_loss,train_targets=train_n,test_targets=test_n,generation=generation,
        training_sha256=base.digest(path),checkpoint_sha256=r['checkpoint_sha256'],
        selection_sha256=base.digest(Path(output)/'selection.json'),completed=True,
        evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    base.save_json(target,result);print('EVALUATED',dataset,config['id'],seed,flush=True);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--phase',choices=['freeze','train','evaluate','all'],default='all')
    args=ap.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    frozen=plan();path=output/'plan.json'
    if path.exists():
        stored=json.loads(path.read_text())
        if any(stored[k]!=v for k,v in frozen.items()):raise ValueError('Plan/code/data changed; use a new output')
        frozen=stored
    else:
        frozen['frozen_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();base.save_json(path,frozen)
    if args.phase=='freeze':print('FROZEN',len(CONFIGS)*12,'combinations',flush=True);return
    jobs=[(d,c,s) for d in base.DATASETS for c in CONFIGS for s in base.SEEDS]
    if args.phase in ['train','all']:
        done=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(train_one,*job,output,frozen) for job in jobs]
            for future in as_completed(futures):
                r=future.result();done.append(dict(dataset=r['dataset'],config=r['config']['id'],seed=r['seed']))
                base.save_json(output/'training_index.json',done)
        print('TRAINING COMPLETE',len(done),flush=True)
    if args.phase in ['evaluate','all']:
        selection=lock_selection(output,frozen);done=[]
        print('VALIDATION SELECTION LOCKED',selection['selected'],flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(evaluate_one,*job,output,frozen,selection['selected']) for job in jobs]
            for future in as_completed(futures):
                r=future.result();done.append(dict(dataset=r['dataset'],config=r['config']['id'],seed=r['seed']))
                base.save_json(output/'evaluation_index.json',done)
        print('EVALUATION COMPLETE',len(done),flush=True)
if __name__=='__main__':main()
