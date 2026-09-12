"""Validate the complete corrected experiment and regenerate empirical paper assets."""
from __future__ import annotations
import argparse
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from corrected_experiments import ROOT, DATASETS, SEEDS, code_hash, digest, save_json
from multi_dataset_fair_experiments import get_model_specs

MODELS = [name for name, _, _ in get_model_specs()]
LABELS = ['RNN', 'GRU', 'LSTM', 'Peephole', 'CIFG', 'Transformer']
CORPORA = ['Shakespeare', 'Alice', 'Pride', 'Sherlock']

def mean_sd(values):
    return st.mean(values), st.stdev(values)

def cell(values):
    mean, sd = mean_sd(values)
    return f'${mean:.3f} \\pm {sd:.3f}$'

def table(path, header, rows, caption, label, wide=False):
    env = 'table*' if wide else 'table'
    columns = 'l'+'r'*(len(header)-1)
    text = f'\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n'
    text += '\\begin{tabular}{'+columns+'}\n\\toprule\n'
    text += ' & '.join(header)+' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(row)+' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'+f'\\end{{{env}}}\n'
    path.write_text(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'artifacts/corrected_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/corrected_assets')
    args = parser.parse_args()
    runs = {}
    source_hash = code_hash()
    for dataset in DATASETS:
        for path in (args.results/dataset).glob('*.json'):
            result = json.loads(path.read_text())
            key = (dataset,result['model'],result['seed'])
            if key in runs or result['code_sha256'] != source_hash or not result['completed']:
                raise ValueError(f'Duplicate, stale or incomplete result: {path}')
            if digest(path.parent/result['checkpoint']) != result['checkpoint_sha256']:
                raise ValueError(f'Missing or changed checkpoint: {path}')
            if result['best_val_loss'] != min(h['val_loss'] for h in result['epoch_history']):
                raise ValueError('Checkpoint is not the validation minimum')
            if result['evaluation_tokens'] != result['corpus']['split_sizes'][2]-1:
                raise ValueError('Incomplete test coverage')
            for mode in ['greedy','sampled']:
                samples = result['generation'][mode]
                if len(samples) != 32 or any(len(x['prompt'])!=64 or len(x['continuation'])!=160 or len(x['reference'])!=160 for x in samples):
                    raise ValueError('Generation protocol mismatch')
            runs[key] = result
    expected = {(d,m,s) for d in DATASETS for m in MODELS for s in SEEDS}
    if set(runs) != expected:
        raise ValueError(f'Expected all 72 runs; found {len(runs)}')
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    def losses(dataset, model):
        return [runs[dataset,model,s]['final_test_loss'] for s in SEEDS]
    def generation(model, mode, metric):
        # Average prompts within corpus, then corpora within each training seed.
        return [st.mean(st.mean(x['metrics'][metric] for x in runs[d,model,s]['generation'][mode]) for d in DATASETS) for s in SEEDS]
    rows=[]
    for d,label in zip(DATASETS,CORPORA):
        c=runs[d,MODELS[0],SEEDS[0]]['corpus']
        rows.append([label,str(c['characters']),str(c['vocabulary']),*[str(n) for n in c['split_sizes']]])
    table(out/'corpora.tex',['Corpus','Characters','Alphabet','Train','Validation','Test'],rows,
          'Frozen corpus inventory and sequential partition sizes, in characters.','tab:corpora',True)
    rows=[]
    for m,label,width in zip(MODELS,LABELS,['125','73','62','62','73','48']):
        p=[runs[d,m,SEEDS[0]]['params'] for d in DATASETS]
        rows.append([label,width,f'{min(p):,}--{max(p):,}'])
    table(out/'models.tex',['Model','Width','Parameters'],rows,
          'One recurrent layer or one Transformer block. Width denotes recurrent hidden size or Transformer model dimension. Parameter ranges reflect corpus alphabets.','tab:models')
    table(out/'loss.tex',['Model']+CORPORA,[[l]+[cell(losses(d,m)) for d in DATASETS] for m,l in zip(MODELS,LABELS)],
          'Test negative log likelihood in nats per character; mean $\\pm$ sample standard deviation across three training seeds. Each held-out target is scored once.','tab:loss',True)
    rows=[]
    for m,label in zip(MODELS,LABELS):
        rows.append([label]+[cell(generation(m,mode,metric)) for mode in ['greedy','sampled'] for metric in ['char_trigram_f1','repetition_rate_4gram']])
    table(out/'generation.tex',['Model','Greedy F1','Greedy repetition','Sampled F1','Sampled repetition'],rows,
          'Surface diagnostics averaged over 32 prompts per corpus and four corpora, then summarized across three seeds. F1 is character trigram overlap; repetition is the fraction of repeated character four-gram occurrences. Sampling temperature is 0.8.','tab:generation',True)
    plt.rcParams.update({'font.size':10,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,4,figsize=(7.15,2.8),sharey=True,layout='constrained')
    for ax,d,title in zip(axes,DATASETS,CORPORA):
        values=[mean_sd(losses(d,m)) for m in MODELS]
        ax.errorbar([v[0] for v in values],range(6),xerr=[v[1] for v in values],fmt='o',capsize=3)
        ax.set_yticks(range(6),LABELS);ax.set_title(title);ax.set_xlabel('Test NLL (nats)');ax.grid(axis='x',alpha=.25)
    axes[0].invert_yaxis();fig.savefig(out/'loss.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.2),layout='constrained')
    for ax,mode in zip(axes,['greedy','sampled']):
        for i,(m,label) in enumerate(zip(MODELS,LABELS)):
            x,xerr=mean_sd(generation(m,mode,'repetition_rate_4gram'))
            y,yerr=mean_sd(generation(m,mode,'char_trigram_f1'))
            ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',capsize=2,label=label,color=f'C{i}')
        ax.set_title(mode.capitalize());ax.set_xlabel('Four-gram repetition');ax.set_ylabel('Trigram F1');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8,loc='best');fig.savefig(out/'generation.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.7),layout='constrained')
    for ax,d,title in zip(axes.flat,DATASETS,CORPORA):
        for i,(m,label) in enumerate(zip(MODELS,LABELS)):
            for j,s in enumerate(SEEDS):
                h=runs[d,m,s]['epoch_history'];ax.plot([v['epoch'] for v in h],[v['val_loss'] for v in h],color=f'C{i}',alpha=.65,linewidth=.8,label=label if j==0 else None)
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Validation NLL');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);fig.savefig(out/'learning.png');plt.close(fig)
    macro={m:st.mean(st.mean(losses(d,m)) for d in DATASETS) for m in MODELS}
    ranked=sorted(macro,key=macro.get)
    best=ranked[0];best_label=LABELS[MODELS.index(best)]
    stops={reason:sum(r['stop_reason']==reason for r in runs.values()) for reason in ['validation_plateau','max_epochs_reached']}
    summary=dict(runs=len(runs),source_hash=source_hash,macro_nll=macro,stop_counts=stops,
                 dataset_winners={d:LABELS[MODELS.index(min(MODELS,key=lambda m:st.mean(losses(d,m))))] for d in DATASETS},
                 generation={m:{mode:{metric:mean_sd(generation(m,mode,metric)) for metric in ['char_trigram_f1','repetition_rate_4gram']} for mode in ['greedy','sampled']} for m in MODELS})
    macros={'BestModel':best_label,'BestNLL':f'{macro[best]:.3f}','TransformerNLL':f'{macro[MODELS[-1]]:.3f}',
            'PlateauCount':str(stops['validation_plateau']),'BudgetCount':str(stops['max_epochs_reached']),
            'RunnerUpModel':LABELS[MODELS.index(ranked[1])], 'RunnerUpNLL':f'{macro[ranked[1]]:.3f}',
            'RankGap':f'{macro[ranked[1]]-macro[best]:.4f}',
            'TransformerGreedyRep':f"{st.mean(generation(MODELS[-1],'greedy','repetition_rate_4gram')):.3f}",
            'TransformerSampledRep':f"{st.mean(generation(MODELS[-1],'sampled','repetition_rate_4gram')):.3f}"}
    (out/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    timing=args.results/'inference_timing.json'
    if timing.exists():
        t=json.loads(timing.read_text());rows=[]
        for m,label in zip(MODELS,LABELS):
            vals=[]
            for mode in ['window64','full_history_state']:
                speeds=[r['median_chars_per_second'] for r in t['rows'] if r['model']==m and r['mode']==mode]
                vals.append(f'{st.mean(speeds):.0f}' if speeds else '--')
            rows.append([label]+vals)
        table(out/'timing.tex',['Model','Window 64','Stateful'],rows,
              'Serial CPU generation throughput (characters/s), batch size one: mean across three seeds of the median of three timed continuations. The two columns use different context semantics and are not a controlled architecture comparison.','tab:timing')
        summary['timing']=t
    save_json(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['generation','timing']},indent=2))

if __name__=='__main__':
    main()
