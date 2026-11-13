"""Small immutable files and the archive's exact identity/hash conventions."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STRATA=[f'{i}to{i+10}' for i in range(10,100,10)]
BUDGETS=[2,4,8,16,32,64,128,256,'FULL']

def require(condition,message):
    if not condition: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def array_sha(x):
    x=np.ascontiguousarray(x);h=hashlib.sha256(json.dumps([x.shape,x.dtype.str]).encode());h.update(x.tobytes());return h.hexdigest()

def source_array_sha(x,experiment):
    """Preserve the two original encodings rather than reinterpreting a hash."""
    if experiment!='lung_ood':return array_sha(x)
    x=np.ascontiguousarray(x)
    header=json.dumps([list(x.shape),str(x.dtype)],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(header+x.tobytes()).hexdigest()

def pair_sha(pairs):
    return hashlib.sha256('\n'.join(f'{c}|{g}' for c,g in pairs).encode()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x));f.write('\n')

def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.save(f,x,allow_pickle=False)
    return dict(path=str(path.resolve()),sha256=sha(path),array_sha256=array_sha(x),shape=list(x.shape),dtype=str(x.dtype))

def load(d):
    require(sha(d['path'])==d['sha256'],'Changed array file')
    x=np.load(d['path'],mmap_mode='r',allow_pickle=False)
    require(list(x.shape)==d['shape'] and str(x.dtype)==d['dtype'] and array_sha(x)==d['array_sha256'] and np.isfinite(x).all(),'Changed or nonfinite numerical array')
    return x
