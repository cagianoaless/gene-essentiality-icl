#!/usr/bin/env python3
"""Score model predictions or aggregate the released experiment metrics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from icl.io import ROOT, read, write, require, load, sha, array_sha, pair_sha
from icl.evaluation import METRICS, aggregate, paired, canonical, metrics


def summarize(out):
    out=Path(out)
    require(not out.exists(), "Output already exists")
    frame=canonical(pd.read_csv(ROOT/"expected/unit_metrics.csv", float_precision="round_trip"))
    out.mkdir(parents=True)
    endpoints, strata=aggregate(frame)
    endpoints.to_csv(out/"endpoints.csv",index=False)
    strata.to_csv(out/"stratum_means.csv",index=False)
    comparisons=[]
    for dataset,left,right in [
        ("depmap","tabpfn","rf"),
        ("lung_ood","tabpfn","rf"),
        ("kumo_E","kumo16","kumo1"),
    ]:
        group=frame[frame.experiment.eq(dataset)]
        if len(group):
            comparisons.append(paired(group,left,right))
    if comparisons:
        deltas=pd.concat(comparisons,ignore_index=True)
        deltas.to_csv(out/"paired_unit_deltas.csv",index=False)
        aggregate(deltas)[0].to_csv(out/"paired_endpoints.csv",index=False)
    write(out/"summary.json",{"units":int(len(frame)),"metrics":METRICS})


def score_new(prepared,result,truth,out):
    """Prediction validation precedes any held-out numeric truth read."""
    r=read(result);m=read(prepared);require(r['status']=='ok' and r['manifest_sha256']==sha(prepared) and not r['query_truth_accessed'],'Invalid prediction receipt')
    c=next(c for c in m['contexts'] if c['context_id']==r['context_id']);p=load(r['prediction']);require(p.shape==(c['n_query'],) and c['query_sha256']==r['query_sha256'],'Wrong query support')
    require(r['context_sha256']==c['context_sha256'] and r['input_hashes']=={k:c[k]['array_sha256'] for k in ['X','Q','y']},'Wrong trained context')
    require(sha(c['query_identity_file'])==c['query_identity_file_sha256'],'Query identity file changed')
    ids=pd.read_csv(c['query_identity_file']);pairs=pd.MultiIndex.from_frame(ids[['ModelID','gene']]);require(pair_sha(pairs)==c['query_sha256'],'Query rows reordered')
    if str(truth).endswith('.npy'):y=np.load(truth,allow_pickle=False)
    else:
        from icl.data import training_targets,targets_at
        if m['config']['experiment']!='synthetic':require(sha(truth)==next(d['sha256'] for d in read(ROOT/'data/sources.json')['inputs'] if d['path']=='raw/depmap-gene-dependencies.csv'),'Wrong raw target snapshot')
        table=training_targets(truth,sorted(ids.ModelID.unique()),sorted(ids.gene.unique()),parser='python' if m['config']['experiment']=='lung_ood' else 'pandas');y=targets_at(table,pairs)
    if m['config']['experiment']!='synthetic':
        qexp='lung_ood' if m['config']['experiment']=='lung_ood' else 'depmap';qid=f'{qexp}/fold{c["fold"]}/{c["stratum"]}';reference=read(ROOT/'expected/query_manifest.json')[qid]
        require(reference['query_sha256']==c['query_sha256'] and array_sha(y)==reference['targets_sha256'],'Query targets differ from original aligned values')
    write(out,dict(metrics=metrics(y,p),query_sha256=r['query_sha256'],
                   original_feature_hash_verified=r.get('original_feature_hash_verified'),
                   scoring_only_target_access=True))



if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--expected",action="store_true",help="Aggregate bundled experimental metrics")
    parser.add_argument("--prepared")
    parser.add_argument("--prediction")
    parser.add_argument("--truth")
    args=parser.parse_args()
    if args.expected:
        require(not any([args.prepared,args.prediction,args.truth]),"Choose aggregation or prediction scoring")
        summarize(args.out)
    else:
        require(all([args.prepared,args.prediction,args.truth]),"Provide --prepared, --prediction and --truth")
        score_new(args.prepared,args.prediction,args.truth,args.out)
