"""Deterministic matched-context character LM experiments.

Run from any directory. Results/checkpoints are tied to code, corpus and configuration
hashes; an incomplete or incompatible run is never silently reused.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
from torch.nn import functional as F
from multi_dataset_fair_experiments import get_model_specs
from generation_quality_benchmark import evaluate_generation

ROOT=Path(__file__).resolve().parent
DATASETS=['tinyshakespeare','alice_in_wonderland','pride_and_prejudice','sherlock_holmes']
SEEDS=[7,11,19]
CONFIG=dict(version=1,sequence_length=64,batch_size=32,steps_per_epoch=100,max_epochs=50,
            patience=5,min_delta=.002,learning_rate=.001,prompts=32,continuation=160,
            temperature=.8,threads=1,alphabet='closed_corpus',evaluation='full_disjoint_blocks')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def code_hash():
    files=sorted([ROOT/'corrected_experiments.py', ROOT/'generation_quality_benchmark.py', ROOT/'multi_dataset_fair_experiments.py', ROOT/'tiny_transformer.py', *ROOT.glob('primitive_*.py')])
    return hashlib.sha256(''.join(f.name+digest(f) for f in files).encode()).hexdigest()
def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def load_corpus(dataset):
    path=ROOT/('tinyshakespeare.txt' if dataset=='tinyshakespeare' else 'datasets/'+dataset+'.txt')
    text=path.read_text();vocab=sorted(set(text));stoi={c:i for i,c in enumerate(vocab)}
    ids=torch.tensor([stoi[c] for c in text],dtype=torch.long);n=len(ids);a=int(.9*n);b=int(.95*n)
    return (ids[:a],ids[a:b],ids[b:]),vocab,dict(dataset=dataset,sha256=digest(path),characters=n,
        vocabulary=len(vocab),split_sizes=[a,b-a,n-b],heldout_only_characters=sorted(set(text[a:])-set(text[:a])))
def batch(stream,length,size,generator):
    if len(stream)<=length:raise ValueError('Stream needs at least L+1 tokens')
    starts=torch.randint(len(stream)-length,(size,),generator=generator)
    idx=starts[:,None]+torch.arange(length)[None,:]
    return stream[idx],stream[idx+1]
@torch.no_grad()
def evaluate(model,stream,length=64,size=32):
    """Score every target except the first exactly once, resetting each block."""
    was_training=model.training;model.eval();total=0.;count=0
    full=(len(stream)-1)//length
    for first in range(0,full,size):
        starts=torch.arange(first,min(first+size,full))*length
        ix=starts[:,None]+torch.arange(length)[None,:]
        x,y=stream[ix],stream[ix+1];z=model(x)
        total+=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),reduction='sum').item();count+=y.numel()
    start=full*length
    if start<len(stream)-1:
        z=model(stream[start:-1][None,:]);y=stream[start+1:]
        total+=F.cross_entropy(z[0],y,reduction='sum').item();count+=len(y)
    model.train(was_training)
    return total/count,count

def select_prompts(test,length,continuation,number):
    # Nonoverlapping held-out blocks; sample blocks with a separate deterministic RNG.
    width=length+continuation;starts=list(range(0,len(test)-width+1,width))
    random.Random(8128).shuffle(starts);starts=sorted(starts[:number])
    if len(starts)<number:raise ValueError('Not enough nonoverlapping prompt/reference blocks')
    return starts,torch.stack([test[s:s+length] for s in starts]),torch.stack([test[s+length:s+width] for s in starts])
@torch.no_grad()
def generate(model,prompts,length,temperature=None,seed=1234,context=64):
    """Equal finite context for all architectures; CPU generator independent of training."""
    was_training=model.training;model.eval();ids=prompts.clone();rng=torch.Generator().manual_seed(seed)
    for _ in range(length):
        logits=model(ids[:,-context:])[:,-1,:]
        nxt=logits.argmax(-1) if temperature is None else torch.multinomial(torch.softmax(logits/temperature,-1),1,generator=rng).squeeze(-1)
        ids=torch.cat([ids,nxt[:,None]],1)
    model.train(was_training);return ids[:,prompts.size(1):]
@torch.no_grad()
def generate_stateful(model,prompts,length,temperature=None,seed=1234):
    """Optional recurrent deployment path; full-history state, NOT fixed-window equivalent."""
    if not hasattr(model,'backbone'):raise ValueError('Stateful path requires a recurrent model')
    was_training=model.training;model.eval();emb=model.embedding(prompts);_,state=model.backbone(emb)
    hidden=state[0] if isinstance(state,tuple) else state;out=[];rng=torch.Generator().manual_seed(seed)
    for i in range(length):
        logits=model.output(hidden)
        nxt=logits.argmax(-1) if temperature is None else torch.multinomial(torch.softmax(logits/temperature,-1),1,generator=rng).squeeze(-1)
        out.append(nxt)
        if i+1<length:
            _,state=model.backbone(model.embedding(nxt[:,None]),state)
            hidden=state[0] if isinstance(state,tuple) else state
    model.train(was_training);return torch.stack(out,1)
def checkpoint_decision(loss,best,patience_best,stale,cfg):
    save=loss<best
    if save:best=loss
    if loss<patience_best-cfg['min_delta']:patience_best=loss;stale=0
    else:stale+=1
    return best,patience_best,stale,save

def train_job(dataset,model_name,seed,output,cfg,source_hash,resume):
    torch.set_num_threads(cfg['threads']);torch.manual_seed(seed);random.seed(seed)
    (train,val,test),vocab,metadata=load_corpus(dataset)
    key=model_name.lower().replace(' ','_')+f'_seed_{seed}'
    directory=Path(output)/dataset;directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/(key+'.json');checkpoint=directory/(key+'.pt')
    signature=hashlib.sha256(json.dumps(dict(config=cfg,corpus=metadata['sha256'],code=source_hash,model=model_name,seed=seed),sort_keys=True).encode()).hexdigest()
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if not resume:raise RuntimeError(f'{result_path} exists; use --resume or a new output directory')
        if old.get('signature')!=signature:raise RuntimeError('Configuration/code/data changed; use a new output directory')
        if checkpoint.exists() and digest(checkpoint)==old.get('checkpoint_sha256'):return old
    factory=next(f for name,_,f in get_model_specs() if name==model_name)
    model=factory(len(vocab));optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    rng=torch.Generator().manual_seed(seed+100000);best=float('inf');patience_best=float('inf');stale=0;history=[]
    started=time.perf_counter();stop='max_epochs_reached'
    for epoch in range(1,cfg['max_epochs']+1):
        model.train();total=0.
        for step in range(cfg['steps_per_epoch']):
            x,y=batch(train,cfg['sequence_length'],cfg['batch_size'],rng)
            optimizer.zero_grad(set_to_none=True);z=model(x);loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1))
            if not bool(torch.isfinite(loss)):raise RuntimeError('Non-finite training loss')
            loss.backward();optimizer.step();total+=loss.item()
        val_loss,val_tokens=evaluate(model,val,cfg['sequence_length'])
        best,patience_best,stale,save=checkpoint_decision(val_loss,best,patience_best,stale,cfg)
        if save:
            tmp=checkpoint.with_suffix('.pt.tmp');torch.save(model.state_dict(),tmp);tmp.replace(checkpoint);best_epoch=epoch
        history.append(dict(epoch=epoch,train_loss=total/cfg['steps_per_epoch'],val_loss=val_loss))
        if epoch%10==0:print(dataset,model_name,seed,'epoch',epoch,'val',round(val_loss,4),flush=True)
        if stale>=cfg['patience']:stop='validation_plateau';break
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    test_loss,test_tokens=evaluate(model,test,cfg['sequence_length'])
    starts,prompts,refs=select_prompts(test,cfg['sequence_length'],cfg['continuation'],cfg['prompts'])
    generation={};decode=lambda row:''.join(vocab[int(t)] for t in row)
    for mode,temp in [('greedy',None),('sampled',cfg['temperature'])]:
        sequences=generate(model,prompts,cfg['continuation'],temp,seed+200000,cfg['sequence_length'])
        generation[mode]=[dict(start_index=s,prompt=decode(p),reference=decode(ref),continuation=decode(seq),metrics=evaluate_generation(decode(seq),decode(ref))) for s,p,ref,seq in zip(starts,prompts,refs,sequences)]
    result=dict(signature=signature,code_sha256=source_hash,config=cfg,corpus=metadata,model=model_name,seed=seed,
        params=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,best_val_loss=best,
        final_test_loss=test_loss,test_ppl=float(torch.exp(torch.tensor(test_loss,dtype=torch.float64))),
        evaluation_tokens=test_tokens,stop_reason=stop,completed=True,epoch_history=history,
        generation=generation,checkpoint=checkpoint.name,checkpoint_sha256=digest(checkpoint),
        elapsed_seconds=time.perf_counter()-started,device='cpu',python=platform.python_version(),torch=torch.__version__,
        platform=platform.platform(),timing_scope='parallel job elapsed; not comparative efficiency evidence')
    save_json(result_path,result);print('DONE',dataset,model_name,seed,round(test_loss,4),flush=True);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'artifacts/corrected_v1');ap.add_argument('--resume',action='store_true');ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--dataset',choices=DATASETS,action='append');ap.add_argument('--model',action='append');ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    args=ap.parse_args();output=args.output.resolve();models=[n for n,_,_ in get_model_specs() if not args.model or n in args.model]
    if not models:ap.error('No matching models')
    jobs=[(d,m,s) for d in (args.dataset or DATASETS) for m in models for s in args.seeds];source_hash=code_hash();results=[]
    config_path=output/'config.json'
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        if not args.resume or previous['config']!=CONFIG or previous['source_hash']!=source_hash:
            raise RuntimeError('Output exists or protocol changed; use --resume with identical protocol or a fresh directory')
    save_json(config_path,dict(config=CONFIG,jobs=jobs,source_hash=source_hash))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending=[executor.submit(train_job,*j,output,CONFIG,source_hash,args.resume) for j in jobs]
        for future in as_completed(pending):
            results.append(future.result());save_json(output/'index.json',[dict(dataset=r['corpus']['dataset'],model=r['model'],seed=r['seed'],signature=r['signature']) for r in results])
    print('COMPLETE',len(results),'runs',flush=True)
if __name__=='__main__':main()
