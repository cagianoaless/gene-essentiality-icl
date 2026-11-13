"""Stage only authorized, exact frozen assets; never downloads or approximates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()

def stage(source,root=ROOT,manifest=None):
    source=Path(source).resolve();root=Path(root).resolve()
    registry=Path(manifest) if manifest else root/"data/private_assets.json"
    records=json.loads(registry.read_text())["files"]
    work=[]
    for rel,entry in records.items():
        src=(source/rel).resolve();dst=(root/rel).resolve()
        if not src.is_relative_to(source) or not dst.is_relative_to(root):
            raise ValueError("Unsafe asset path")
        if not src.is_file() or digest(src)!=entry["sha256"]:
            raise ValueError("Missing or changed authorized asset: "+rel)
        if dst.exists() and digest(dst)!=entry["sha256"]:
            raise ValueError("Refusing to overwrite a different asset: "+rel)
        work.append((src,dst))
    # Validate the complete incoming bundle before copying any file.
    for src,dst in work:
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    return {"staged_or_already_identical":len(work),"downloads":0}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",required=True)
    a=p.parse_args();print(stage(a.source))
