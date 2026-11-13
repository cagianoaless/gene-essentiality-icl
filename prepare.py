#!/usr/bin/env python3
"""Validate released data and prepare one or all immutable experiment contexts."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from icl.io import ROOT,STRATA,BUDGETS,require,read,sha,array_sha,source_array_sha,pair_sha,write,save
from icl.data import read_counts,training_targets,targets_at,masks,cohorts,trajectories_checked,expression_state,feature_matrix


def prepare(config,data_root,out,fold=None,stratum=None,budget=None,seed=None,mode=None):
    config_path=Path(config);cfg=yaml.safe_load(config_path.read_text());exp=cfg['experiment'];out=Path(out).resolve()
    require(not out.exists(),'Never overwrite prepared scientific inputs')
    data_root=Path(data_root).resolve();sources=read(ROOT/'data/sources.json')['inputs']
    for d in sources:
        require(sha(data_root/d['path'])==d['sha256'],'Wrong input snapshot: '+d['path'])
    counts,cells,stable=read_counts(data_root/'counts.h5ad');meta=pd.read_csv(ROOT/'data/metadata/expression_mapping.csv')
    require(len(cells)==661 and counts.shape==(661,17172),'Wrong expression dimensions')
    require(np.array_equal(stable[meta.expression_column],meta.stable_id),'Expression stable-ID mapping differs')
    mapping=dict(zip(meta.gene,meta.expression_column));pair_masks=masks();frozen=read(ROOT/'expected/contexts.json');prepared=[];states={}
    allgenes=sorted(mapping);out.mkdir(parents=True)
    is_lung=exp=='lung_ood';cohort_exp='lung_ood' if is_lung else 'depmap'
    fold_ids=cfg['folds'] if fold is None else [fold]
    for f in fold_ids:
        train,test=cohorts(cohort_exp,f);require(set(cells)==set(train)|set(test) and not set(train)&set(test),'Cohort mismatch')
        orders=trajectories_checked(cohort_exp,f,train,test,pair_masks) if exp!='kumo_E' else None
        skey=f'{cohort_exp}/fold{f}';scores,norm,loading,report=expression_state(counts,cells,train,test,skey,mode or cfg['pca_mode'],**cfg['preprocessing_tolerances'])
        states[skey]=report
        # Only this fold's training numeric targets are parsed.
        labels=training_targets(data_root/'raw/depmap-gene-dependencies.csv',train,allgenes,parser='python' if is_lung else 'pandas')
        for st in ([stratum] if stratum else cfg['strata']):
            mask=pair_masks[st];pool=mask[mask.get_level_values(0).isin(train)];query=mask[mask.get_level_values(0).isin(test)]
            if is_lung:rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261027,6,STRATA.index(st)])))
            else:rng=np.random.default_rng(f)
            full=pool[rng.permutation(len(pool))]
            Q=feature_matrix(query,cells,scores,norm,loading,mapping,cfg['representation']);qdesc=save(out/f'fold{f}/{st}/Q.npy',Q)
            query_file=out/f'fold{f}/{st}/query_ids.csv';pd.DataFrame(query.tolist(),columns=['ModelID','gene']).to_csv(query_file,index=False)
            budgets=cfg['budgets'] if budget is None else [budget]
            for b in budgets:
                seeds=[None] if b=='FULL' else ([0] if exp=='kumo_E' else (cfg['selection_seeds'] if seed is None else [seed]))
                for s in seeds:
                    if exp=='kumo_E':context=full[:32768];cid=f'kumo_E/fold{f}/{st}/C32768'
                    else:
                        selected=train if b=='FULL' else orders[s][:int(b)];context=full[full.get_level_values(0).isin(selected)]
                        require(len(context)==len(pool[pool.get_level_values(0).isin(selected)]),'Selected eligible pairs omitted')
                        base='lung_ood' if is_lung else 'depmap';prefix='' if is_lung else f'fold{f}/';label='shared' if b=='FULL' else str(s)
                        cid=f'{base}/{prefix}{st}/selection{label}/C{b}'
                    require(cid in frozen,'Unregistered context: '+cid);expected=frozen[cid]
                    require(pair_sha(context)==expected['context_sha256'] and pair_sha(query)==expected['query_sha256'],'Pair identity/order changed')
                    require(not set(context.get_level_values(0))&set(query.get_level_values(0)),'Train/query overlap')
                    panel=sorted(set(mask.get_level_values(1)));gene_index={g:i for i,g in enumerate(panel)}
                    genes=np.array([gene_index[g] for g in context.get_level_values(1)],dtype=np.int32)
                    require((np.bincount(genes,minlength=len(panel))>0).all(),'Incomplete gene coverage')
                    X=feature_matrix(context,cells,scores,norm,loading,mapping,cfg['representation']);y=targets_at(labels,context)
                    require(X.shape==(expected['n_train'],cfg['feature_width']) and len(Q)==expected['n_query'],'Unexpected feature dimensions')
                    require(source_array_sha(y,cohort_exp)==expected['y_sha256'],'Training label values or order changed')
                    if (mode or cfg['pca_mode'])=='frozen':
                        require(source_array_sha(X,cohort_exp)==expected['X_sha256'] and source_array_sha(Q,cohort_exp)==expected['Q_sha256'],'Historical model-interface feature values changed')
                    dest=out/cid
                    record=dict(context_id=cid,experiment=exp,fold=f,stratum=st,budget=b,seed=s,n_cells=len(set(context.get_level_values(0))),n_train=len(y),n_query=len(Q),
                        context_sha256=pair_sha(context),query_sha256=pair_sha(query),X=save(dest/'X.npy',X),Q=qdesc,y=save(dest/'y.npy',y),
                        context_genes=save(dest/'context_genes.npy',genes),query_genes=save(dest/'query_genes.npy',np.array([gene_index[g] for g in query.get_level_values(1)],dtype=np.int32)),
                        gene_count=len(panel),query_identity_file=str(query_file),query_identity_file_sha256=sha(query_file),query_truth_accessed=False)
                    prepared.append(record)
        print('Prepared fold',f,'contexts',len(prepared),flush=True)
    require(prepared,'No jobs selected');write(out/'manifest.json',dict(config=cfg,config_sha256=sha(config_path),contexts=prepared,preprocessing=states,query_truth_accessed=False))
    return out/'manifest.json'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--data-root');p.add_argument('--out',required=True);p.add_argument('--fold',type=int);p.add_argument('--stratum',choices=STRATA);p.add_argument('--budget');p.add_argument('--seed',type=int,choices=range(5));p.add_argument('--pca-mode',choices=['frozen','refit']);a=p.parse_args()
    require(a.config and a.data_root,'--config and --data-root are required')
    b=None if a.budget is None else 'FULL' if a.budget=='FULL' else int(a.budget)
    print(prepare(a.config,a.data_root,a.out,a.fold,a.stratum,b,a.seed,a.pca_mode))
