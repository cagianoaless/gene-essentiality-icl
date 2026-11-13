#!/usr/bin/env python3
"""Paper figures: verified five-seed endpoints; shared FULL appears only once."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icl.io import ROOT,require,write
from icl.evaluation import canonical,aggregate

COLORS={'tabpfn':'#1764aa','rf':'#d56b17','gene_mean':'#777777'}
LABELS={'tabpfn':'TabPFN-3.5','rf':'Random Forest','gene_mean':'Gene mean'}

def plot(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    units=canonical(pd.read_csv(ROOT/'expected/unit_metrics.csv',float_precision='round_trip'));endpoints,_=aggregate(units)
    plt.rcParams.update({'font.size':20,'axes.labelsize':20,'xtick.labelsize':20,'ytick.labelsize':20,'legend.fontsize':20,'pdf.fonttype':42,'ps.fonttype':42})
    receipt=[]
    for dataset,filename in [('depmap','depmap_seed_auroc'),('lung_ood','lung_ood_seed_auroc')]:
        fig,ax=plt.subplots(figsize=(16,9));group=endpoints[endpoints.experiment.eq(dataset)]
        full_counts=units[(units.experiment==dataset)&units.budget.eq('FULL')].n_cells
        lo,hi=float(full_counts.min()),float(full_counts.max());full_x=(lo+hi)/2
        for model in COLORS:
            g=group[group.model.eq(model)];finite=g[~g.budget.eq('FULL')]
            require(len(finite)==40 and set(finite.seed)==set(range(5)),'Need exactly five seeds at each finite budget')
            for seed in range(5):
                line=finite[finite.seed.eq(seed)].copy();line['count']=line.budget.astype(int);line=line.sort_values('count')
                ax.plot(line['count'],line.auroc,color=COLORS[model],alpha=.18,linewidth=2)
            means=finite.assign(count=finite.budget.astype(int)).groupby('count').auroc.mean().sort_index();full=g[g.budget.eq('FULL')]
            require(len(full)==1,'FULL is one shared endpoint, not five seed observations')
            ax.plot(list(means.index)+[full_x],list(means)+[float(full.iloc[0].auroc)],color=COLORS[model],linewidth=1.5,marker='o',markersize=5,label=LABELS[model])
            if lo!=hi:ax.errorbar([full_x],[float(full.iloc[0].auroc)],xerr=[(hi-lo)/2],color=COLORS[model],capsize=3,linewidth=1)
        ticks=[2,4,8,16,32,64,128,256,full_x];labels=[str(x) for x in ticks[:-1]]+[f'{int(lo)}–{int(hi)}\nFULL' if lo!=hi else f'{int(lo)}\nFULL']
        ax.set_xscale('log',base=2);ax.set_xticks(ticks,labels);ax.set_xlabel('#cell-lines');ax.set_ylabel('AUROC');ax.set_title('');ax.grid(alpha=.15);ax.legend(frameon=False,loc='upper left',ncol=3)
        fig.subplots_adjust(left=.09,right=.98,top=.96,bottom=.18)
        for suffix in ['pdf','png']:
            path=out/(filename+'.'+suffix);require(not path.exists(),'Do not overwrite an existing manuscript figure');fig.savefig(path,dpi=180,bbox_inches=None)
        receipt.append(dict(experiment=dataset,figure_size_inches=list(fig.get_size_inches()),font_size=20,no_title=True,individual_seed_alpha=.18,mean_linewidth=1.5,seed_FULL_replicates=0,FULL_cell_count_range=[int(lo),int(hi)],note='DepMap horizontal FULL span shows actual fold sizes, not an uncertainty interval. Lung FULL is555.'))
        plt.close(fig)
    write(out/'plot_receipt.json',receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='figures');a=p.parse_args();plot(a.out)
