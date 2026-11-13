"""Metrics and aggregation for the DepMap and Lung experiments."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, average_precision_score
from .io import STRATA, require

METRICS=['auroc','r2','auprc']

def metrics(y,p):
    require(y.shape==p.shape and y.ndim==1 and np.isfinite(y).all() and np.isfinite(p).all(),'Target/prediction mismatch')
    two=len(np.unique(y>.5))==2
    return dict(auroc=float(roc_auc_score(y>.5,p)) if two else None,
                r2=float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else None,
                auprc=float(average_precision_score(y>.5,p)) if (y>.5).any() else None)

def canonical(frame):
    frame=frame.copy();frame['budget']=frame.budget.astype(str).str.replace(r'\.0$','',regex=True)
    frame['seed']=pd.to_numeric(frame.seed,errors='coerce');frame.loc[frame.budget.eq('FULL'),'seed']=np.nan
    return frame

def aggregate(frame,diagnostic44=False):
    frame=canonical(frame);rows=[];vectors=[]
    for (exp,model,budget,seed),g in frame.groupby(['experiment','model','budget','seed'],dropna=False,sort=False):
        if exp=='version_audit':continue  # Three diagnostic units never form a benchmark headline.
        current=g[~((g.fold==0)&g.stratum.eq('60to70'))] if diagnostic44 and exp!='lung_ood' else g
        require(not current.duplicated(['fold','stratum']).any(),'Duplicate evaluation unit')
        expected={(f,s) for f in (range(5) if exp in ['depmap','kumo_E'] else [0]) for s in STRATA
                  if not (diagnostic44 and exp!='lung_ood' and (f,s)==(0,'60to70'))}
        complete=set(zip(current.fold,current.stratum))==expected
        v=current.groupby('stratum')[METRICS].mean().reindex(STRATA)
        info=dict(experiment=exp,model=model,budget=budget,seed=None if pd.isna(seed) else int(seed),analysis='inherited44_diagnostic' if diagnostic44 else 'primary',units=len(current),complete=complete)
        row={**info}
        for metric in METRICS:
            # Undefined metrics cannot disappear through pandas skipna.
            defined=complete and current[metric].notna().all() and v[metric].notna().all()
            row[metric]=float(v[metric].median()) if defined else None
            row[metric+'_mean_strata']=float(v[metric].mean()) if defined else None
        rows.append(row)
        for st in STRATA:vectors.append({**info,'stratum':st,**{m:float(v.loc[st,m]) if pd.notna(v.loc[st,m]) else None for m in METRICS}})
    return pd.DataFrame(rows),pd.DataFrame(vectors)

def paired(frame,left='tabpfn',right='rf'):
    frame=canonical(frame);keys=['experiment','fold','stratum','budget','seed'];a=frame[frame.model.eq(left)];b=frame[frame.model.eq(right)]
    require(not a.duplicated(keys).any() and not b.duplicated(keys).any(),'Duplicate paired unit')
    g=a.merge(b,on=keys,how='outer',suffixes=('_left','_right'),indicator=True,validate='one_to_one')
    require(g._merge.eq('both').all(),'Paired comparison has unmatched units')
    require((g.query_sha256_left==g.query_sha256_right).all() and (g.context_sha256_left==g.context_sha256_right).all(),'Comparison mixes different supports')
    out=g[keys].copy();out['model']=left+'-minus-'+right
    for m in METRICS:out[m]=g[m+'_left']-g[m+'_right']
    return out
