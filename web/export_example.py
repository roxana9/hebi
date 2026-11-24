#!/usr/bin/env python3
"""
Run HEBI once through the web app and save the finished run as static files,
so the page can show it without a server (Hugging Face Static Space).

Run on a machine with HEBI and a GPU, next to app.py:

    HEBI_VOCAB=/path/default_gene_vocab.json \
    HEBI_CHECKPOINT=/path/hebi_hybrid_oldL0.pt \
    python export_example.py /path/xenium_breast_cancer_rep1.h5ad \
        --mode continual --epochs 20 --niche-objective spatial --n-metis 4096 \
        --out example

Then upload the folder as data/example/ in the Space (next to index.html).
Genes are stored as 8-bit values with one scale per gene (max), which is
enough for colouring cells and keeps the files small.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5ad", nargs="+", help="one or more .h5ad sections")
    p.add_argument("--out", default="example")
    p.add_argument("--mode", default="continual", choices=["zero_shot", "continual"])
    p.add_argument("--init", default="pretrained", choices=["pretrained", "random"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--niche-objective", default="original", choices=["original", "spatial"])
    p.add_argument("--n-metis", type=int, default=1024)
    p.add_argument("--species", default="Human")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--n-hvg", type=int, default=0)
    p.add_argument("--normalise", default="false", choices=["true", "false"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from fastapi.testclient import TestClient
    import app as web

    c = TestClient(web.app)
    out = Path(a.out)
    for sub in ("clusters", "niches", "genes"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # 1. run
    params = {"mode": a.mode, "init": a.init, "epochs": a.epochs, "lr": a.lr,
              "niche_objective": a.niche_objective, "n_metis": a.n_metis, "species": a.species,
              "n_clusters": a.n_clusters, "n_hvg": a.n_hvg, "normalise": a.normalise, "seed": a.seed}
    r = c.post("/api/jobs", params=params)
    r.raise_for_status()
    jid = r.json()["job_id"]
    for f in a.h5ad:
        print(f"uploading {f}")
        with open(f, "rb") as fh:
            c.post(f"/api/jobs/{jid}/files", params={"filename": Path(f).name}, content=fh.read()).raise_for_status()
    c.post(f"/api/jobs/{jid}/start").raise_for_status()
    shown = 0
    while True:
        s = c.get(f"/api/jobs/{jid}").json()
        log = s.get("log") or []
        for line in log[shown:]:
            print("  " + line)
        shown = len(log)
        if s["status"] in ("done", "error"):
            break
        time.sleep(3)
    if s["status"] == "error":
        sys.exit(f"run failed: {s.get('error')}")
    base = f"/api/jobs/{jid}"

    def save(path, url, **q):
        r = c.get(base + url, params=q)
        r.raise_for_status()
        (out / path).write_bytes(r.content)
        return r

    # 2. the run itself
    meta = c.get(base + "/meta").json()
    save("coords.f32", "/coords")
    save("sections.u8", "/sections")
    save("emb_index.u32", "/embedding/index")
    save("emb_xy.f32", "/embedding/xy")

    print(f"clusters k = {meta['min_k']}..{meta['max_k']}")
    for k in range(meta["min_k"], meta["max_k"] + 1):
        (out / f"clusters/k{k}.json").write_text(json.dumps(c.get(base + "/clusters", params={"k": k}).json()))
        save(f"clusters/k{k}.u8", "/labels", k=k)

    if meta.get("has_niches"):
        ks = meta.get("niche_ks") or {}
        ks = ks if isinstance(ks, dict) else {"keep": ks}
        todo = [(0, "keep")] + [(k, pol) for pol, lst in ks.items() for k in lst]
        print(f"niches: {len(todo)} resolutions")
        for k, pol in todo:
            (out / f"niches/{pol}_k{k}.json").write_text(
                json.dumps(c.get(base + "/niches", params={"k": k, "policy": pol}).json()))
            save(f"niches/{pol}_k{k}.u8", "/niche_labels", k=k, policy=pol)

    # 3. genes, 8-bit with one scale per gene
    scale = {}
    genes = meta["genes"]
    print(f"genes: {len(genes)}")
    for g in genes:
        v = np.frombuffer(c.get(base + "/gene", params={"name": g}).content, dtype=np.float32)
        m = float(v.max()) if v.size else 0.0
        scale[g] = m if m > 0 else 1.0
        q = np.clip(np.rint(v / scale[g] * 255), 0, 255).astype(np.uint8)
        (out / "genes" / f"{g}.u8").write_bytes(q.tobytes())

    meta.update(example=True, gene_scale=scale)
    (out / "meta.json").write_text(json.dumps(meta))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"saved {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
