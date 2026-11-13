"""Target-free identities/features and a row-filtered training-target reader."""
import csv
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from .io import ROOT,STRATA,BUDGETS,require,read,sha,array_sha,source_array_sha,pair_sha
from .preprocessing import ExpressionPCA
from .selection import trajectories


def h5column(group,name):
    require(name!='score','Official artifact score access is forbidden')
    x=group[name]
    if isinstance(x,h5py.Dataset):return x.asstr()[:] if x.dtype.kind in 'OSU' else x[:]
    code=x['codes'][:];require((code>=0).all(),'Missing categorical ID')
    return x['categories'].asstr()[:][code]


def read_counts(path):
    with h5py.File(path,'r') as h:
        cells=h5column(h['obs'],'ModelID').astype(str)
        stable=h5column(h['var'],h['var'].attrs['_index']).astype(str)
        x=h['X']
        if isinstance(x,h5py.Group):
            require(x.attrs['encoding-type']=='csr_matrix','Unsupported count storage')
            counts=sparse.csr_matrix((x['data'][:],x['indices'][:],x['indptr'][:]),shape=tuple(x.attrs['shape'])).toarray()
        else:counts=x[:]
    counts=counts.astype(np.float32)
    require(len(set(cells))==len(cells) and len(set(stable))==len(stable) and np.isfinite(counts).all() and (counts>=0).all(),'Invalid counts/IDs')
    return counts,pd.Index(cells),pd.Index(stable)


def training_targets(path,cells,genes,parser='pandas'):
    """Preserve each study's exact parser; neither parses held-out numeric fields."""
    if parser=='pandas':
        # Verbatim numerical behavior of the original DepMap reader.
        with Path(path).open() as handle:header=next(csv.reader(handle))
        names=[s.split(' ')[0] for s in header[1:]];columns=[]
        for gene in genes:
            matches=[i+1 for i,name in enumerate(names) if name==gene]
            require(len(matches)==1,'Ambiguous dependency header: '+gene);columns.append(matches[0])
        identities=pd.read_csv(path,usecols=[0],dtype=str).iloc[:,0]
        selected=np.flatnonzero(identities.isin(cells));selected_rows=set((selected+1).tolist())
        frame=pd.read_csv(path,usecols=[0,*columns],skiprows=lambda row:row>0 and row not in selected_rows,index_col=0)
        frame.columns=[s.split(' ')[0] for s in frame.columns]
        require(frame.index.is_unique and set(frame.index)==set(cells),'Missing/duplicate training ID')
        require(all(pd.api.types.is_numeric_dtype(frame[g]) for g in frame.columns),'Nonnumeric selected training target')
        return frame.loc[list(cells),list(genes)]
    require(parser=='python','Unknown historical parser')
    wanted=set(cells);rows={}
    with Path(path).open(newline='') as f:
        header=next(csv.reader([f.readline()]));symbols=[s.split(' ')[0] for s in header[1:]]
        require(len(symbols)==len(set(symbols)),'Ambiguous target symbols')
        index={g:i+1 for i,g in enumerate(symbols)};require(set(genes)<=set(index),'Missing target gene')
        for raw in f:
            model_id=raw.partition(',')[0].strip().strip('"')
            if model_id not in wanted:continue
            fields=next(csv.reader([raw]));require(len(fields)==len(header) and fields[0]==model_id and model_id not in rows,'Malformed/duplicate target row')
            rows[model_id]=[float(fields[index[g]]) if fields[index[g]] else np.nan for g in genes]
    require(set(rows)==wanted,'Missing training ModelID')
    return pd.DataFrame.from_dict(rows,orient='index',columns=genes).loc[cells]


def targets_at(table,pairs):
    row=table.index.get_indexer(pairs.get_level_values(0));col=table.columns.get_indexer(pairs.get_level_values(1))
    require((row>=0).all() and (col>=0).all(),'Target identity missing')
    y=table.to_numpy()[row,col].astype(np.float64);require(np.isfinite(y).all(),'Do not silently drop an eligible pair')
    return y


def masks():
    for name,expected in read(ROOT/'data/metadata/manifest.json').items():require(sha(ROOT/'data/metadata'/name)==expected,'Protocol metadata changed: '+name)
    result={}
    for s in STRATA:
        frame=pd.read_parquet(ROOT/f'data/metadata/examples-{s}.parquet',columns=['cell-line','gene']).astype(str)
        pairs=pd.MultiIndex.from_frame(frame)
        require(pairs.is_unique and pairs.equals(pairs.sort_values()),'Released identity order changed')
        result[s]=pairs
    return result


def cohorts(experiment,fold):
    if experiment=='lung_ood':
        m=read(ROOT/'data/metadata/selections.json')['lung_ood'];return m['train_cells'],m['test_cells']
    s=pd.read_csv(ROOT/'data/metadata/splits.csv')
    return sorted(s.loc[s[f'fold-{fold}']=='train','cell-line']),sorted(s.loc[s[f'fold-{fold}']=='val','cell-line'])


def trajectories_checked(experiment,fold,train,test,pair_masks):
    frozen=read(ROOT/'data/metadata/selections.json')[experiment];genes=sorted({g for p in pair_masks.values() for g in p.get_level_values(1)})
    require(len(genes)==2566,'Benchmark gene panel changed')
    columns={};row={c:i for i,c in enumerate(train)}
    for st,pairs in pair_masks.items():
        panel=sorted(set(pairs.get_level_values(1)));a=np.zeros((len(train),len(panel)),dtype=bool);gi={g:j for j,g in enumerate(panel)}
        for c,g in pairs:
            if c in row:a[row[c],gi[g]]=True
        for g,j in gi.items():
            if g in columns:require(np.array_equal(columns[g],a[:,j]),'Shared-gene eligibility mismatch')
            columns[g]=a[:,j]
    eligible=np.column_stack([columns[g] for g in genes])
    if experiment=='depmap':
        valid,records=trajectories(train,eligible,fold)
        require(len(valid)==frozen['qualifying_C2_pairs'][str(fold)],'C2 qualifying count changed')
        orders=[r['ordered_trajectory'] for r in records]
    else:
        # Exact original Lung RNG namespaces and list ordering.
        import itertools
        bits=[sum(1<<int(j) for j in np.flatnonzero(a)) for a in eligible];target=(1<<len(genes))-1
        valid=[(train[i],train[j]) for i,j in itertools.combinations(range(len(train)),2) if bits[i]|bits[j]==target]
        require(len(valid)==58 and all('ACH-000682' in p for p in valid),'Lung C2 anchor certificate changed')
        rng=lambda namespace,*c:np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261027,namespace,*c])))
        choices=rng(1).choice(len(valid),size=5,replace=False);orders=[]
        for seed,choice in enumerate(choices):
            first=list(valid[int(choice)]);remaining=[c for c in train if c not in first];orders.append(first+rng(2,seed).permutation(remaining).tolist())
    expected=[r['ordered_cells'] for r in frozen['trajectories'] if r['fold']==fold]
    require(orders==expected,'Cell selection differs from archived five trajectories')
    for order in orders:
        previous=set()
        for budget in BUDGETS:
            selected=train if budget=='FULL' else order[:budget]
            require(previous<set(selected) and not set(selected)&set(test),'Nestedness/leakage violation')
            require(eligible[[row[c] for c in selected]].any(axis=0).all(),'Incomplete aggregate gene coverage')
            previous=set(selected)
    return orders


def restore_components(model,strides):
    """Preserve the archived oversampled F-order view used by float32 BLAS."""
    if list(model.components_.strides)!=list(strides):
        item=model.components_.dtype.itemsize;leading=strides[1]//item
        require(strides[0]==item and leading>=15,'Unrecognized historical component layout')
        backing=np.empty((leading,model.components_.shape[1]),dtype=model.components_.dtype,order='F');backing[:15]=model.components_;model.components_=backing[:15]


def expression_state(counts,cells,train,test,key,mode='frozen',atol=5e-4,rtol=1e-5):
    metadata_path = ROOT/'data/metadata/preprocessing.json'
    if not metadata_path.exists():
        require(mode=='refit',
                'Original PCA states are unavailable; use --pca-mode refit or supply the frozen states')
        ri=cells.get_indexer(train)
        require((ri>=0).all(), 'Missing training expression cell')
        pca=ExpressionPCA(15,0).fit(counts[ri],train)
        pca.assert_training_only(train,test)
        scores=np.empty((len(cells),15),dtype=np.float32)
        for ids in [train,test]:
            ix=cells.get_indexer(ids)
            scores[ix]=pca.transform(counts[ix])
        norm=pca.normalize(counts)
        loading=pca.gene_loadings().astype(np.float32)
        report=dict(mode='refit',reference_available=False,fit_cells=train,test_cells=test,
                    target_sum=pca.target_sum_,query_truth_accessed=False)
        return scores,norm,loading,report
    meta=read(metadata_path)[key];require(meta['fit_cells']==train and meta['test_cells']==test,'Wrong PCA cohort')
    path=ROOT/meta['path'];require(sha(path)==meta['sha256'],'Reference coefficients changed')
    with np.load(path,allow_pickle=False) as h:reference={k:h[k].copy() for k in h.files}
    ri=cells.get_indexer(train);require((ri>=0).all(),'Missing expression cell')
    if mode=='refit':
        pca=ExpressionPCA(15,0).fit(counts[ri],train)
        for name,value in reference.items():require(np.allclose(getattr(pca.model_,name+'_'),value,atol=atol,rtol=rtol),'PCA refit outside declared tolerance: '+name)
    else:
        require(mode=='frozen','Use frozen or refit preprocessing')
        pca=ExpressionPCA(15,0);pca.target_sum_=meta['target_sum'];pca.fit_ids_=tuple(train);pca.model_=PCA(**meta['pca_parameters'])
        for name,value in reference.items():setattr(pca.model_,name+'_',value)
        pca.model_.n_features_in_=counts.shape[1]
    pca.assert_training_only(train,test);restore_components(pca.model_,meta['component_strides'])
    scores=np.empty((len(cells),15),dtype=np.float32)
    for ids in [train,test]:
        ix=cells.get_indexer(ids);scores[ix]=pca.transform(counts[ix])
    norm=pca.normalize(counts);loading=pca.gene_loadings().astype(np.float32)
    report=dict(mode=mode,fit_cells=train,test_cells=test,target_sum=pca.target_sum_,component_strides=list(pca.model_.components_.strides),
                coefficients_exact=all(np.array_equal(getattr(pca.model_,k+'_'),v) for k,v in reference.items()),
                scores_exact=source_array_sha(scores,key.split('/')[0])==meta['scores_sha256'],loading_exact=source_array_sha(loading,key.split('/')[0])==meta['loading_sha256'],
                normalized_exact=('normalized_sha256' not in meta or array_sha(norm)==meta['normalized_sha256']),atol=atol,rtol=rtol)
    if mode=='frozen':require(report['scores_exact'] and report['loading_exact'] and report['normalized_exact'],'Reconstructed frozen state differs; inspect environment, do not silently replace')
    return scores,norm,loading,report


def feature_matrix(pairs,cells,scores,norm,loading,mapping,representation):
    ri=cells.get_indexer(pairs.get_level_values(0));ci=np.asarray([mapping[g] for g in pairs.get_level_values(1)])
    require((ri>=0).all() and (ci>=0).all(),'Feature identity unresolved')
    blocks=[scores[ri],loading[ci]]
    if representation=='F':blocks.append(norm[ri,ci,None])
    x=np.column_stack(blocks).astype(np.float32);require(np.isfinite(x).all() and x.shape[1]==(31 if representation=='F' else 30),'Feature interface mismatch')
    return x
