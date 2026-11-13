#!/usr/bin/env python3
"""Real model APIs; only sanitized training labels and target-free queries are read."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import argparse
import importlib.metadata as metadata
from pathlib import Path
import resource
import subprocess
import sys
import time
import warnings
import numpy as np
import pandas as pd
from icl.io import ROOT,require,read,sha,load,save,source_array_sha,write


def implementation_hashes():
    return {name: sha(ROOT/name) for name in ['predict.py', 'icl/models.py', 'icl/io.py']}


def validate_completed(manifest, context, model_name, result, checkpoint_sha256=None):
    """Validate a reusable output without fitting a model or reading query truth."""
    m = read(manifest); r = read(result)
    require(r.get('status') == 'ok' and r.get('model') == model_name,
            'Cannot reuse an incomplete or different model result')
    require(r.get('manifest_sha256') == sha(manifest)
            and r.get('context_id') == context['context_id'], 'Cannot reuse different inputs')
    for field in ['context_sha256', 'query_sha256', 'n_train', 'n_query']:
        require(r.get(field) == context[field], 'Completed output has wrong '+field)
    require(r.get('query_truth_accessed') is False, 'Invalid inference truth-access receipt')
    require(r.get('implementation_sha256') == implementation_hashes(),
            'Execution code changed since this output was produced')
    require(r.get('input_hashes') == {k:context[k]['array_sha256'] for k in ['X','Q','y']},
            'Completed output has different feature/label hashes')
    for key in ['X','Q','y']:
        load(context[key])
    if model_name in ['tabpfn','kumo1','kumo16']:
        expected = read(ROOT/'results/main/checkpoints.json')[
            'tabpfn' if model_name == 'tabpfn' else 'kumo']['sha256']
        require(checkpoint_sha256 == expected == r.get('checkpoint_sha256'),
                'Checkpoint identity required for reuse')
        if model_name == 'tabpfn':
            require(r.get('package_version') == m['config']['tabpfn_version'],
                    'TabPFN package differs from the recorded condition')
    p = load(r['prediction'])
    require(p.shape == (context['n_query'],), 'Completed prediction shape changed')
    return r


def predict(manifest,context_id,model_name,out,checkpoint=None):
    manifest=Path(manifest);m=read(manifest);cfg=m['config'];matches=[c for c in m['contexts'] if c['context_id']==context_id];require(len(matches)==1,'Context ID is not unique');c=matches[0]
    require(model_name in cfg.get('models',['gene_mean','rf']),'Model is outside the selected experiment')
    out=Path(out);require(not out.exists(),'Never overwrite an attempt or silently retry');out.mkdir(parents=True);start=time.perf_counter()
    record=dict(model=model_name,context_id=context_id,manifest_sha256=sha(manifest),context_sha256=c['context_sha256'],query_sha256=c['query_sha256'],query_truth_accessed=False,n_train=c['n_train'],n_query=c['n_query'],input_hashes={k:c[k]['array_sha256'] for k in ['X','Q','y']},implementation_sha256=implementation_hashes())
    write(out/'started.json',record)
    X,Q,y=[load(c[k]) for k in ['X','Q','y']];require(X.shape==(c['n_train'],cfg['feature_width']) and Q.shape==(c['n_query'],cfg['feature_width']),'Wrong feature interface')
    require(y.shape==(len(X),),'Training labels misaligned');t=time.perf_counter();trace=[]
    if cfg['experiment']!='synthetic':
        original=read(ROOT/'expected/contexts.json')[c['context_id']]
        scheme='lung_ood' if cfg['experiment']=='lung_ood' else 'depmap'
        refit_without_reference=any(
            report.get('reference_available') is False
            for report in m.get('preprocessing',{}).values())
        blocks=[('y',y)] if refit_without_reference else [('X',X),('Q',Q),('y',y)]
        require(all(source_array_sha(x,scheme)==original[name+'_sha256'] for name,x in blocks),
                'Prepared arrays differ from the original benchmark')
        record['original_feature_hash_verified']=not refit_without_reference
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if model_name=='gene_mean':
            g,qg=[load(c[k]) for k in ['context_genes','query_genes']];count=np.bincount(g,minlength=c['gene_count']);require((count>0).all(),'No gene-mean fallback is permitted')
            prediction=(np.bincount(g,weights=y,minlength=c['gene_count'])/count)[qg];record['fit_seconds']=0.;record['prediction_seconds']=time.perf_counter()-t
        elif model_name=='rf':
            from sklearn.ensemble import RandomForestRegressor
            require(metadata.version('scikit-learn')=='1.7.2','Use the pinned RF environment')
            model=RandomForestRegressor(**cfg['rf_parameters']);model.fit(X,y);record['fit_seconds']=time.perf_counter()-t;t=time.perf_counter();prediction=model.predict(Q);record['prediction_seconds']=time.perf_counter()-t;record['parameters']=model.get_params()
        else:
            require(checkpoint,'An explicit existing checkpoint is required; automatic downloads are disabled')
            from icl.models import make_model,model_execution,predict_model
            cp=read(ROOT/'results/main/checkpoints.json')['kumo' if model_name.startswith('kumo') else 'tabpfn'];require(sha(checkpoint)==cp['sha256'],'Wrong checkpoint')
            import torch
            require(torch.cuda.is_available(),'CUDA unavailable; no scientific CPU fallback');torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
            if model_name=='tabpfn':
                require(metadata.version('tabpfn')==cfg['tabpfn_version'],'TabPFN version differs from selected condition')
                model=make_model('tabpfn',device='cuda',seed=0,n_estimators=8,checkpoint=checkpoint,tabpfn_options=cfg['execution'],categorical_features_indices=[])
                # Observe the actual helper used by fit/predict, without changing
                # arguments, numerical outputs, allocation or RNG state.
                import tabpfn.inference as ti
                original=ti.should_save_peak_mem
                def observed(*args,**kwargs):
                    flag=original(*args,**kwargs);trace.append(dict(arguments={k:str(v) for k,v in kwargs.items()},decision=bool(flag)));return flag
                ti.should_save_peak_mem=observed
                try:
                    columns=[f'feature_{i}' for i in range(cfg['feature_width'])]
                    with model_execution(model):model.fit(pd.DataFrame(X,columns=columns),y)
                    torch.cuda.synchronize();record['fit_seconds']=time.perf_counter()-t
                    require(model.n_estimators_==8 and model.n_train_samples_==len(y) and model.inference_config_.SUBSAMPLE_SAMPLES is None and model.ensemble_preprocessor_.subsample_row_indices is None,'Hidden context subsampling')
                    require(list(model.categorical_features_indices_ or [])==[],'Declared feature modalities changed')
                    shapes=[list(cache.train_shape) for cache in model.executor_.kv_caches];require(all(s[1]==len(y) for s in shapes) and sum(s[0] for s in shapes)==8,'Incomplete estimator caches')
                    from tabpfn.preprocessing.datamodel import FeatureModality
                    from dataclasses import asdict
                    record.update(cache_train_shapes=shapes,inferred_categories=model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL),inference_config=asdict(model.inference_config_),package_version=metadata.version('tabpfn'))
                    t=time.perf_counter();prediction=predict_model(model,pd.DataFrame(Q,columns=columns),8192)
                finally:ti.should_save_peak_mem=original
            else:
                require(model_name in ['kumo1','kumo16'],'Unsupported model')
                require(metadata.version('structured-data-models')=='0.1.0.dev0','Use the pinned NVIDIA source revision')
                import sdm
                package=Path(sdm.__file__).resolve().parent
                source_registry=ROOT/'data/metadata/sdm_source_hashes.json'
                if source_registry.exists():
                    for name,expected in read(source_registry)['sources'].items():
                        require(sha(package/name)==expected,'Installed NVIDIA source differs from pinned revision')
                model=make_model('kumo',device='cuda',seed=0,n_estimators=1 if model_name=='kumo1' else 16,checkpoint=checkpoint,kumo_size='large',categorical_features_indices=[])
                columns=[f'feature_{i}' for i in range(30)];model.fit(pd.DataFrame(X,columns=columns),y);torch.cuda.synchronize();record['fit_seconds']=time.perf_counter()-t;t=time.perf_counter()
                prediction=predict_model(model,pd.DataFrame(Q,columns=columns),1024)
            torch.cuda.synchronize();record.update(prediction_seconds=time.perf_counter()-t,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),GPU=torch.cuda.get_device_name(0),checkpoint_sha256=cp['sha256'],memory_heuristic_observations=trace)
    prediction=np.asarray(prediction);require(prediction.shape==(c['n_query'],) and np.isfinite(prediction).all(),'Invalid prediction vector')
    record.update(status='ok',prediction=save(out/'predictions.npy',prediction),total_seconds=time.perf_counter()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,warnings=[str(w.message) for w in caught]);write(out/'result.json',record)
    return out/'result.json'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--context');p.add_argument('--model',choices=['tabpfn','rf','gene_mean','kumo1','kumo16','all'],required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint');p.add_argument('--all',action='store_true');a=p.parse_args()
    if a.all:
        m=read(a.prepared);models=m['config'].get('models',['gene_mean']) if a.model=='all' else [a.model]
        checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None
        for c in m['contexts']:
            for model in models:
                out=Path(a.out)/model/c['context_id'];r=out/'result.json'
                if r.exists():
                    validate_completed(a.prepared,c,model,r,checkpoint_sha256);continue
                cmd=[sys.executable,str(Path(__file__).resolve()),'--prepared',a.prepared,'--context',c['context_id'],'--model',model,'--out',str(out)]
                if a.checkpoint:cmd+=['--checkpoint',a.checkpoint]
                subprocess.run(cmd,check=True,timeout=86400 if model.startswith('kumo') else 7200)
    else:
        require(a.context and a.model!='all','Select --context or --all');print(predict(a.prepared,a.context,a.model,a.out,a.checkpoint))
