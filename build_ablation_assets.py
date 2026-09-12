"""Report the full factorial grid and paired intervention effects."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ablation_experiments as exp
from ablation_models import configurations, is_original
import corrected_experiments as base
from build_paper_results_assets import table

CORPUS_LABELS=['Shakespeare','Alice','Pride','Sherlock']
ARCH_LABELS={'transformer':'Transformer','rnn':'RNN','gru':'GRU'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=exp.OUTPUT)
    ap.add_argument('--output',type=Path,default=exp.ROOT/'artifacts/ablation_assets')
    args=ap.parse_args();root=args.results.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((root/'plan.json').read_text());selection=exp.lock_selection(root,frozen)
    training=exp.collect(root,frozen)
    records={}
    for _,r in training:
        c=r['config'];p=root/'evaluation'/r['dataset']/(exp.key(c,r['seed'])+'.json')
        x=json.loads(p.read_text())
        assert x['selection_sha256']==base.digest(root/'selection.json') and x['completed']
        records[r['dataset'],c['id'],r['seed']]=x
    assert len(records)==216
    configs=configurations();tf=[c for c in configs if c['architecture']=='transformer']
    def series(config_id,metric):
        return [st.mean(records[d,config_id,s][metric] for d in base.DATASETS) for s in base.SEEDS]
    def summary(values):return dict(mean=st.mean(values),sd=st.stdev(values),seed_values=values)
    def cell(values):return f'${st.mean(values):.3f} \\pm {st.stdev(values):.3f}$'
    grid={c['id']:{k:summary(series(c['id'],k)) for k in ['train_loss','val_loss','test_loss']} for c in configs}
    rows=[]
    for c in tf:
        rows.append([str(c['width']),str(c['dropout']),f"{c['learning_rate']:g}"]+[cell(series(c['id'],k)) for k in ['train_loss','val_loss','test_loss']])
    table(out/'ablation_grid.tex',['Width','Dropout','Learning rate','Train NLL','Validation NLL','Test NLL'],rows,
          'Complete Transformer factorial grid. Scores average the four corpora equally, then report mean $\\pm$ sample standard deviation across three seed summaries. Width 48 uses feed-forward width 112; width 32 uses 296--309 to preserve the corpus-specific parameter budget. Lower NLL is better.','tab:ablation_grid',True)
    # Paired effects are averaged within a corpus and seed, before seed uncertainty.
    lookup={(c['width'],c['dropout'],c['learning_rate']):c['id'] for c in tf}
    pairs={
        'dropout':[(lookup[w,0.,lr],lookup[w,.1,lr]) for w,lr in itertools.product([48,32],[.0003,.001,.003])],
        'lr_lower':[(lookup[w,p,.0003],lookup[w,p,.001]) for w,p in itertools.product([48,32],[0.,.1])],
        'lr_higher':[(lookup[w,p,.003],lookup[w,p,.001]) for w,p in itertools.product([48,32],[0.,.1])],
        'allocation':[(lookup[32,p,lr],lookup[48,p,lr]) for p,lr in itertools.product([0.,.1],[.0003,.001,.003])]}
    labels={'dropout':'Dropout 0 minus 0.1','lr_lower':'LR 0.0003 minus 0.001','lr_higher':'LR 0.003 minus 0.001','allocation':'Width 32 minus width 48'}
    effects={}
    for name,matched in pairs.items():
        by_corpus={d:[st.mean(records[d,a,s]['test_loss']-records[d,b,s]['test_loss'] for a,b in matched) for s in base.SEEDS] for d in base.DATASETS}
        macro=[st.mean(by_corpus[d][i] for d in base.DATASETS) for i in range(3)]
        effects[name]=dict(label=labels[name],macro=summary(macro),corpora={d:summary(v) for d,v in by_corpus.items()},pairs=matched)
    # Retain conditional effects to avoid masking factor interactions.
    conditional={}
    for w,lr in itertools.product([48,32],[.0003,.001,.003]):
        vals=[st.mean(records[d,lookup[w,0.,lr],s]['test_loss']-records[d,lookup[w,.1,lr],s]['test_loss'] for d in base.DATASETS) for s in base.SEEDS]
        conditional[f'dropout_w{w}_lr{lr}']=summary(vals)
    for p,lr in itertools.product([0.,.1],[.0003,.001,.003]):
        vals=[st.mean(records[d,lookup[32,p,lr],s]['test_loss']-records[d,lookup[48,p,lr],s]['test_loss'] for d in base.DATASETS) for s in base.SEEDS]
        conditional[f'allocation_p{p}_lr{lr}']=summary(vals)
    interactions={}
    # Difference of paired effects at the two allocations or two learning rates.
    for lr in [.0003,.001,.003]:
        a=conditional[f'dropout_w32_lr{lr}']['seed_values']
        b=conditional[f'dropout_w48_lr{lr}']['seed_values']
        interactions[f'dropout_by_allocation_lr{lr}']=summary([x-y for x,y in zip(a,b)])
    for w in [48,32]:
        for lr in [.0003,.003]:
            a=conditional[f'dropout_w{w}_lr{lr}']['seed_values']
            b=conditional[f'dropout_w{w}_lr{.001}']['seed_values']
            interactions[f'dropout_by_learning_rate_w{w}_lr{lr}']=summary([x-y for x,y in zip(a,b)])
    for p in [0.,.1]:
        for lr in [.0003,.003]:
            a=conditional[f'allocation_p{p}_lr{lr}']['seed_values']
            b=conditional[f'allocation_p{p}_lr{.001}']['seed_values']
            interactions[f'allocation_by_learning_rate_p{p}_lr{lr}']=summary([x-y for x,y in zip(a,b)])
    generation={};generation_rows=[]
    for architecture,config_id in selection['selected'].items():
        generation[architecture]={};row=[ARCH_LABELS[architecture]]
        for mode in ['greedy','sampled']:
            generation[architecture][mode]={}
            for metric in ['char_trigram_f1','repetition_rate_4gram']:
                values=[st.mean(st.mean(x['metrics'][metric] for x in records[d,config_id,seed]['generation'][mode]) for d in base.DATASETS) for seed in base.SEEDS]
                generation[architecture][mode][metric]=summary(values);row.append(cell(values))
        generation_rows.append(row)
    table(out/'ablation_generation.tex',['Model','Greedy F1','Greedy repetition','Sampled F1','Sampled repetition'],generation_rows,
          'Generation for globally validation-selected follow-up configurations, using the original prompts and temperature 0.8. Means and sample standard deviations summarize three corpus-averaged training seeds.','tab:ablation_generation',True)
    plt.rcParams.update({'font.size':10,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.5),layout='constrained')
    for ax,(name,effect) in zip(axes.flat,effects.items()):
        vals=[effect['corpora'][d] for d in base.DATASETS]
        ax.axvline(0,color='0.4',linewidth=.8)
        ax.errorbar([v['mean'] for v in vals],range(4),xerr=[v['sd'] for v in vals],fmt='o',capsize=3,color='C0')
        ax.set_yticks(range(4),CORPUS_LABELS);ax.invert_yaxis();ax.set_title(labels[name],fontsize=10)
        ax.set_xlabel('Paired change in test NLL');ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'ablation_effects.png');plt.close(fig)
    comparisons={};rows=[]
    for architecture in ['rnn','gru','transformer']:
        original=next(c for c in configs if c['architecture']==architecture and is_original(c))
        best=next(c for c in configs if c['id']==selection['selected'][architecture])
        comparisons[architecture]=dict(original=original,selected=best,original_scores=grid[original['id']],selected_scores=grid[best['id']],
            paired_test_change=summary([a-b for a,b in zip(series(best['id'],'test_loss'),series(original['id'],'test_loss'))]))
        for label,c in [('Original',original),('Validation-selected',best)]:
            rows.append([ARCH_LABELS[architecture],label,f"{c['learning_rate']:g}"]+[cell(series(c['id'],k)) for k in ['train_loss','val_loss','test_loss']])
    table(out/'ablation_selected.tex',['Model','Setting','Learning rate','Train NLL','Validation NLL','Test NLL'],rows,
          'Original and globally validation-selected settings. Configuration selection averages validation NLL over all four corpora and three seeds before test evaluation. The Transformer receives twelve candidate settings; each recurrent control receives three.','tab:ablation_selected',True)
    contrasts=[]
    for name,effect in effects.items():contrasts.append([labels[name],cell(effect['macro']['seed_values'])])
    table(out/'ablation_contrasts.tex',['Intervention','Paired test NLL change'],contrasts,
          'Marginal paired effects across the prespecified Transformer grid. Negative differences favor the first-listed level. Standard deviations summarize three seed-level corpus averages, not population confidence intervals.','tab:ablation_contrasts',True)
    train_stop={reason:sum(r['stop_reason']==reason for _,r in training) for reason in ['validation_plateau','max_epochs_reached']}
    macros={
        'AblDropEffect':f"{effects['dropout']['macro']['mean']:.3f}",
        'AblLowerLREffect':f"{effects['lr_lower']['macro']['mean']:.3f}",
        'AblHigherLREffect':f"{effects['lr_higher']['macro']['mean']:.3f}",
        'AblAllocationEffect':f"{effects['allocation']['macro']['mean']:.3f}",
        'AblTFNLL':f"{comparisons['transformer']['selected_scores']['test_loss']['mean']:.3f}",
        'AblRNNNLL':f"{comparisons['rnn']['selected_scores']['test_loss']['mean']:.3f}",
        'AblGRUNLL':f"{comparisons['gru']['selected_scores']['test_loss']['mean']:.3f}",
        'AblPlateauCount':str(train_stop['validation_plateau']),'AblBudgetCount':str(train_stop['max_epochs_reached'])}
    (out/'ablation_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    result=dict(combinations=216,new_runs=180,reused=36,source_hash=frozen['source_hash'],selection=selection['selected'],
                grid=grid,effects=effects,conditional_effects=conditional,interactions=interactions,
                generation=generation,comparisons=comparisons,stop_counts=train_stop)
    base.save_json(out/'summary.json',result)
    print(json.dumps(dict(selection=selection['selected'],effects={k:v['macro'] for k,v in effects.items()},
                         selected_test={a:v['selected_scores']['test_loss'] for a,v in comparisons.items()}),indent=2))
if __name__=='__main__':main()
