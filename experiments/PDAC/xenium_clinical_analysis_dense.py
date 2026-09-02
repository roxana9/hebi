#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:31:48 2026

@author: rokny-tower4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenium_clinical_analysis.py



Note: AnnData objects are built directly from the .pt files, so cell
counts and ordering cannot drift between the two. There is no separate
.h5ad to keep in sync.
"""
import argparse
import glob
import os
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch

from hebi.grn.build import build_hetero_grn
from hebi.grn.build_ppi_codex import shared_curated_edges
from hebi.inference.adapt import continual_adapt
from hebi.inference.cluster import (
    KEY_CELL_CLUSTERS,
    KEY_COMPOSITION_NICHES,
    build_composition_vectors,
    cluster_embedding,
)
from hebi.inference.encode import encode_section, load_hebi
from hebi.training import pretrain
from hebi.training.pretrain import Train

DENSE_TOP_N = 1000  # how many densest genes (by frac cells expressing) for GRN


def set_global_seed(seed: int = 1990):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_random_checkpoint(pt_dir, vocab_path, out_path, device,
                             n_proto_l1=1024, n_proto_l2=20, seed=1990):
    print(f"\n🟡 Building random-init checkpoint (lvl1={n_proto_l1}, lvl2={n_proto_l2})")
    set_global_seed(seed)
    trainer = Train(
        data_dir=pt_dir, vocab_path=vocab_path,
        l0_encoder_type="transformer", batch_size=2048, num_workers=4,
        lr=1e-4, num_epochs=1, global_num_layers=2,
        num_prototypes_lvl1=n_proto_l1, num_prototypes_lvl2=n_proto_l2,
        decoder_hidden_dim=256, early_stopping_patience=10,
        checkpoint_path=None, device=device, seed=seed,
    )
    trainer.discover_data_files()
    trainer.data_path = trainer.data_files[0]
    trainer.load_data()
    trainer.build_models(trainer.X.size(0), trainer.X.shape)
    torch.save({
        "epoch": 0,
        "gene_embedder": trainer.gene_embedder.state_dict(),
        "gene_linear": trainer.gene_linear.state_dict(),
        "model_lvl0": trainer.model_lvl0.state_dict(),
        "model_lvl1": trainer.model_lvl1.state_dict(),
        "pooler_lvl2": trainer.pooler_lvl2.state_dict(),
        "model_lvl2": trainer.model_lvl2.state_dict(),
        "global_decoder": trainer.global_decoder.state_dict(),
    }, out_path)
    print(f"✅ Random-init checkpoint created -> {out_path}")
    del trainer
    torch.cuda.empty_cache()
    return out_path


def infer_n_proto_l1(pt_dir):
    f = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))[0]
    d = torch.load(f, map_location="cpu", weights_only=False)
    sp = d["super_sp"]
    n = sp.shape[-1] if hasattr(sp, "shape") else int(np.asarray(sp).shape[-1])
    print(f"   Data contains {n} METIS partitions -> num_prototypes_lvl1={n}")
    return int(n)


def make_subset_dir(pt_dir, files, tag="subset"):
    sub = os.path.join("/tmp", f"xenium_{tag}_{len(files)}")
    if os.path.exists(sub):
        for f in glob.glob(os.path.join(sub, "*.pt")):
            os.unlink(f)
    os.makedirs(sub, exist_ok=True)
    for f in files:
        dst = os.path.join(sub, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(pt_dir, f), dst)
    print(f"   Subset directory ready: {sub} ({len(files)} samples)")
    return sub


def build_adata_list(pt_dir, files, label_cols):
    adatas, labels = [], []
    for f in files:
        d = torch.load(os.path.join(pt_dir, f), map_location="cpu", weights_only=False)
        X = d["X"]
        X = X.float().numpy() if torch.is_tensor(X) else np.asarray(X, float)
        coords = d["coords"]
        coords = coords.numpy() if torch.is_tensor(coords) else np.asarray(coords)

        a = ad.AnnData(X=X)
        a.var_names = [str(g) for g in d["gene_names"]]
        a.var_names_make_unique()
        a.obsm["spatial"] = np.asarray(coords, dtype=float)

        sid = d.get("sample_id", f.replace(".pt", ""))
        lab = {"sample_id": sid, "patient_id": str(d.get("patient_id", "unknown"))}
        d_labels = d.get("labels", {})
        for c in label_cols:
            val = d_labels.get(c, d.get(c, np.nan))
            lab[c] = str(val) if not pd.isna(val) else "nan"

        adatas.append(a)
        labels.append(lab)
    return adatas, labels


def patch_match_grn(files, gene_graphs_list):
    """Match each sample to its own GRN graph by sample_id.

    The base implementation matches by gene count, which cannot separate
    samples that share a panel — as they do here, since every sample uses
    the same shared dense panel.
    """
    idx_of = {os.path.basename(f).replace(".pt", ""): i for i, f in enumerate(files)}

    def _match(self, slide_path):
        sid = os.path.basename(str(slide_path)).replace(".pt", "")
        i = idx_of.get(sid)
        if i is not None:
            return gene_graphs_list[i]
        print(f"   [warn] No graph for sample {sid}; falling back to graph[0]")
        return gene_graphs_list[0]

    pretrain.Train._match_grn_graph = _match
    print(f"✅ Patched _match_grn_graph -> sample_id ({len(idx_of)} samples mapped)")


def encode_and_cluster(checkpoint_path, pt_dir, files, adata_list, vocab_path,
                       gene_graphs_list, num_domains, comp_num_niches,
                       device, init="pretrained", composition_neighbors=200,
                       seed=1990):
    """Encode every sample, then cluster jointly across the cohort.

    Replaces the old encode_and_cluster_multiple_slides. Prototype merging
    and spatial-domain merging are not used here: only the joint z0
    clustering and the composition niches feed the tissue-level features.
    """
    trainer = load_hebi(
        checkpoint_path=checkpoint_path,
        data_dir=pt_dir,
        vocab_path=vocab_path,
        device=device,
        init=init,
        use_grn=gene_graphs_list is not None,
    )

    all_z0, all_protos, all_grn_attn, counts = [], [], [], []
    for i, f in enumerate(files):
        out = encode_section(
            trainer=trainer,
            section_path=os.path.join(pt_dir, f),
            gene_graph=gene_graphs_list[i] if gene_graphs_list else None,
        )
        all_z0.append(out["z0"].cpu().numpy())
        all_protos.append(out["prototypes"].numpy())
        all_grn_attn.append(out["grn_attention"])
        counts.append(all_z0[-1].shape[0])
        print(f"   Encoded {f}: {counts[-1]} cells")

    joint_z0 = np.concatenate(all_z0, axis=0)
    n_prototypes = int(np.concatenate(all_protos).max()) + 1

    print(f"   Joint KMeans (k={num_domains}) on {joint_z0.shape[0]:,} cells")
    joint_labels = cluster_embedding(
        joint_z0, n_clusters=num_domains, method="kmeans", random_state=seed
    )

    comp_vectors = [
        build_composition_vectors(
            all_protos[i], adata_list[i].obsm["spatial"],
            k=composition_neighbors, n_prototypes=n_prototypes,
        )
        for i in range(len(files))
    ]
    joint_comp = cluster_embedding(
        np.concatenate(comp_vectors, axis=0),
        n_clusters=comp_num_niches, method="kmeans", random_state=seed,
    )
    print(f"   Composition niches: {len(np.unique(joint_comp))}")

    start = 0
    for i, (adata, count) in enumerate(zip(adata_list, counts)):
        sl = slice(start, start + count)
        adata.obsm["z0"] = all_z0[i]
        adata.obs[KEY_CELL_CLUSTERS] = pd.Categorical(
            joint_labels[sl].astype(str))
        adata.obs[KEY_COMPOSITION_NICHES] = pd.Categorical(
            joint_comp[sl].astype(str))
        start += count

    return adata_list, trainer, joint_labels, all_grn_attn


def region_features(attn, edge_index, X, niche_labels, positions, n_niches):
    attn = np.asarray(attn, dtype=float).reshape(-1)
    ei = np.asarray(edge_index)
    X = np.asarray(X, dtype=float)
    nl = np.asarray(niche_labels)
    pos = np.asarray(positions, dtype=int)
    src, dst, a = ei[0, pos], ei[1, pos], attn[pos]
    mat = np.zeros((n_niches, len(pos)), dtype=float)
    for c in range(n_niches):
        m = (nl == c)
        if not m.any():
            continue
        e = X[m].mean(0)
        mat[c] = a * e[src] * e[dst]
    return mat.reshape(-1), mat.mean(0)


def composition_vector(niche_labels, n_niches):
    nl = np.asarray(niche_labels)
    v = np.bincount(nl[nl >= 0], minlength=n_niches).astype(float)
    return v / v.sum() if v.sum() > 0 else v


def zscore(M):
    mu, sd = M.mean(0), M.std(0)
    sd[sd < 1e-8] = 1.0
    return (M - mu) / sd


def save_block(X, labels, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {"X": np.asarray(X, dtype=float), "labels": labels}
    if extra:
        out.update(extra)
    torch.save(out, path)
    print(f"   ✅ Saved {os.path.basename(path):<34} {np.asarray(X).shape}")


def _plot_all_samples(niche_dir, n_niches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    png_dir = os.path.join(niche_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(niche_dir, "*.npz")))

    for f in files:
        sid = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        coords, nl = d["coords"], d["comp_niches"].astype(int)
        resp = str(d["response"]) if "response" in d else "N/A"
        fig, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=nl, cmap="tab20",
                        s=12, alpha=0.85, vmin=0, vmax=max(n_niches - 1, 1),
                        rasterized=True)
        fig.colorbar(sc, ax=ax, label="Cellular Niche", ticks=range(n_niches))
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_title(f"Sample: {sid} | Response: {resp}", fontsize=11)
        ax.axis("off"); fig.tight_layout()
        fig.savefig(os.path.join(png_dir, f"{sid}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"   🖼 Saved {len(files)} spatial PNGs -> {png_dir}")


def main(seed: int = 1990):
    set_global_seed(seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="continual", choices=["zero_shot", "continual"])
    ap.add_argument("--pt_dir", default="/DATA1/Roxana/Hierarchy/xenium_clinical")
    ap.add_argument("--vocab_path", default="/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json")
    ap.add_argument("--checkpoint_path", default="/DATA1/Roxana/Hierarchy/Two_levels/hebi_foundation_best_new.pt")
    ap.add_argument("--init_mode", default="pretrained", choices=["pretrained", "random", "hybrid"])
    ap.add_argument("--out_dir", default="./Xenium_results/clinical_run_dense")
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_domains", type=int, default=20)
    ap.add_argument("--comp_num_niches", type=int, default=12)
    ap.add_argument("--composition_neighbors", type=int, default=200)
    ap.add_argument("--spearman_threshold", type=float, default=0.4)
    ap.add_argument("--species", default="Human")
    ap.add_argument("--max_regions", type=int, default=None)
    ap.add_argument("--label_cols", nargs="+", default=["response", "TRGStage"])
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--dense_top_n", type=int, default=DENSE_TOP_N,
                    help="densest genes (by frac cells expressing) to use for GRN")
    ap.add_argument("--cancer_type", default=None,
                    help="only samples of this cancer type (e.g. CRC)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    label_cols = args.label_cols

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.pt_dir, "*.pt")))
    print(f"📂 Found {len(files)} sample .pt files in {args.pt_dir}")

    work_dir = args.pt_dir

    if args.cancer_type:
        import scanpy as sc
        adata_meta = sc.read_h5ad("/DATA2/Sally/data/xenium_data/adata_xenium.h5ad")
        samp_cancer = adata_meta.obs.groupby("SampleID", observed=True)["CancerType"].first().to_dict()
        del adata_meta
        files = [f for f in files if samp_cancer.get(f.replace(".pt", "")) == args.cancer_type]
        print(f"cancer_type={args.cancer_type}: {len(files)} samples")
        work_dir = make_subset_dir(args.pt_dir, files, tag=args.cancer_type)
    if args.max_regions:
        files = files[:args.max_regions]
        work_dir = make_subset_dir(args.pt_dir, files)

    # ---- 1. BUILD ANNDATA (full HVG panel, for encode/composition) ----
    adata_list, labels = build_adata_list(work_dir, files, label_cols)

    # ---- 2. Determine a SHARED dense-gene panel across ALL samples ----
    # Per-sample dense selection would pick different genes per sample
    # (density varies sample to sample), leaving zero genes in common and
    # therefore zero shared edges across the cohort. Instead: average
    # each gene's "fraction of cells expressing" ACROSS samples, then pick
    # one fixed top-N panel used for every sample's GRN.
    print(f"\n🧬 Selecting a SHARED dense-gene panel (top {args.dense_top_n}) "
          f"across all {len(files)} samples...")
    frac_expr_sum = None
    for adata_i in adata_list:
        Xi = adata_i.X
        Xi = Xi.toarray() if hasattr(Xi, "toarray") else np.asarray(Xi)
        fe = (Xi > 0).mean(0)
        frac_expr_sum = fe if frac_expr_sum is None else frac_expr_sum + fe
    frac_expr_avg = frac_expr_sum / len(adata_list)
    n = min(args.dense_top_n, len(frac_expr_avg))
    shared_order = np.argsort(-frac_expr_avg)
    shared_dense_idx = np.sort(shared_order[:n])
    shared_dense_genes = set(adata_list[0].var_names[i] for i in shared_dense_idx)
    print(f"   shared dense panel: {len(shared_dense_genes)} genes "
          f"(avg min frac_expr={frac_expr_avg[shared_dense_idx].min():.4f})")

    # ---- 3. CONSTRUCT GRNs, all samples using the SAME gene panel ----
    gene_graphs_list, gene_names_list = [], []
    print(f"\n🧬 Constructing per-sample GRNs via build_hetero_grn "
          f"(shared dense panel, {len(shared_dense_genes)} genes)...")
    for i, (f, adata_i) in enumerate(zip(files, adata_list)):
        sid = f.replace(".pt", "")
        keep_idx = [j for j, g in enumerate(adata_i.var_names)
                    if g in shared_dense_genes]
        Xi = adata_i.X
        Xi = Xi.toarray() if hasattr(Xi, "toarray") else np.asarray(Xi)
        a_dense = ad.AnnData(X=Xi[:, keep_idx])
        a_dense.var_names = [adata_i.var_names[j] for j in keep_idx]
        a_dense.var_names_make_unique()
        a_dense.obsm["spatial"] = adata_i.obsm["spatial"]

        print(f"   [{i+1}/{len(files)}] Building GRN for {sid} "
              f"({a_dense.shape[1]} genes, same panel for all samples)...")
        gg, gn, _ = build_hetero_grn(
            adata=a_dense,
            species=args.species,
            spearman_threshold=args.spearman_threshold,
            add_reverse_for_curated=True
        )
        gene_graphs_list.append(gg)
        gene_names_list.append(gn)

    gene_names = gene_names_list[0]
    region_ids = [f.replace(".pt", "") for f in files]

    positions, edge_labels = shared_curated_edges(gene_graphs_list, gene_names)
    n_shared = len(edge_labels)
    print(f"   Shared GRN edges across samples: {n_shared}")
    print(f"   Feature Dim: {args.comp_num_niches} + ({args.comp_num_niches} x {n_shared}) = "
          f"{args.comp_num_niches + args.comp_num_niches * n_shared}")

    patch_match_grn(files, gene_graphs_list)

    # ---- 4. CONTINUAL ADAPTATION ----
    ckpt = os.path.join(args.out_dir, "xenium_continual_grn.pt")
    if args.skip_train and os.path.exists(ckpt):
        print(f"\n⏭ Reusing existing checkpoint: {ckpt}")
    else:
        if args.init_mode == "random":
            n_l1 = infer_n_proto_l1(work_dir)
            start_ckpt = create_random_checkpoint(
                work_dir, args.vocab_path,
                os.path.join(args.out_dir, "xenium_random_init.pt"),
                device, n_proto_l1=n_l1, n_proto_l2=args.num_domains, seed=seed
            )
        else:
            start_ckpt = args.checkpoint_path

        if args.mode == "continual":
            print(f"\n🔄 Continual adaptation — {args.num_epochs} epochs over {len(files)} samples")
            continual_adapt(
                checkpoint_path=start_ckpt,
                data_dir=work_dir,
                vocab_path=args.vocab_path,
                device=device,
                lr=args.lr,
                num_epochs=args.num_epochs,
                output_checkpoint=ckpt,
                gene_graph=None,
                gene_graph_list=gene_graphs_list,
                init=args.init_mode,
            )
        else:
            ckpt = start_ckpt

    # ---- 5. ENCODE & JOINT CLUSTERING (uses FULL adata_list) ----
    print(f"\n🔎 Encoding {len(files)} samples...")
    results, trainer, joint_labels, all_grn_attn = encode_and_cluster(
        checkpoint_path=ckpt,
        pt_dir=work_dir,
        files=files,
        adata_list=adata_list,
        vocab_path=args.vocab_path,
        gene_graphs_list=gene_graphs_list,
        num_domains=args.num_domains,
        comp_num_niches=args.comp_num_niches,
        composition_neighbors=args.composition_neighbors,
        device=device,
        init=args.init_mode,
        seed=seed,
    )

    # ---- 6. TISSUE-LEVEL FEATURE AGGREGATION ----
    print("\n📊 Aggregating tissue-level feature blocks...")
    comp_key = next((k for k in (KEY_COMPOSITION_NICHES, KEY_CELL_CLUSTERS)
                     if k in results[0].obs.columns), None)
    if comp_key is None:
        raise KeyError(f"No niche column found: {list(results[0].obs.columns)}")

    niche_dir = os.path.join(args.out_dir, "sample_niches")
    os.makedirs(niche_dir, exist_ok=True)
    for i, a in enumerate(results):
        sid = labels[i]["sample_id"]
        np.savez(os.path.join(niche_dir, f"{sid}.npz"),
                 coords=np.asarray(a.obsm["spatial"]),
                 comp_niches=a.obs[comp_key].astype(int).values,
                 patient_id=labels[i].get("patient_id", "unknown"),
                 response=labels[i].get("response", "nan"),
                 trg_stage=labels[i].get("TRGStage", "nan"))
    _plot_all_samples(niche_dir, args.comp_num_niches)

    Xc, Xf, Xm, kept = [], [], [], []
    for i, a in enumerate(results):
        attn = all_grn_attn[i]
        if attn is None:
            print(f"   ⚠️ {region_ids[i]}: No attention found, skipping sample")
            continue
        nl = a.obs[comp_key].astype(int).values
        Xdata = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)

        flat, nmean = region_features(
            np.asarray(attn.detach().cpu() if torch.is_tensor(attn) else attn),
            gene_graphs_list[i]["global"].edge_index.cpu().numpy(),
            Xdata, nl, positions[i], args.comp_num_niches
        )
        Xc.append(composition_vector(nl, args.comp_num_niches))
        Xf.append(flat)
        Xm.append(nmean)
        kept.append(labels[i])

    Xc, Xf, Xm = np.stack(Xc), np.stack(Xf), np.stack(Xm)

    o = args.out_dir
    save_block(Xc, kept, f"{o}/xenium_compositiononly.pt")
    save_block(Xf, kept, f"{o}/xenium_edgesonly.pt", {"edge_labels": edge_labels})
    save_block(np.concatenate([zscore(Xc), zscore(Xf)], 1), kept, f"{o}/xenium_fused.pt",
               {"edge_labels": edge_labels, "block_sizes": {"comp": Xc.shape[1], "edges": Xf.shape[1]}})
    save_block(np.concatenate([zscore(Xc), zscore(Xm)], 1), kept, f"{o}/xenium_fused_compact.pt",
               {"edge_labels": edge_labels})

    np.savez(f"{o}/xenium_attention.npz",
             attention=np.stack([
                 np.asarray(a.detach().cpu() if torch.is_tensor(a) else a)[positions[i]]
                 for i, a in enumerate(all_grn_attn) if a is not None]),
             edge_labels=np.array(edge_labels),
             sample_ids=np.array([l["sample_id"] for l in kept]))

    print(f"\n✅ Pipeline Complete -> Outputs saved in {o}")
    return results, all_grn_attn, trainer, gene_graphs_list


if __name__ == "__main__":
    in_ide = any("spyder" in k.lower() or "ipykernel" in k.lower()
                 for k in sys.modules.keys())
    if in_ide:
        print("🖥 IDE mode detected")

    
    PER_SAMPLE = True          # True = train one HEBI model per sample
    CANCER = "pdac"
    PT_DIR = "/DATA1/Roxana/Hierarchy/xenium_clinical_raw"

    base_argv = [
        "xenium_clinical_analysis_dense.py",
        "--mode",              "continual",
        "--pt_dir",            PT_DIR,
        "--vocab_path",        "/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json",
        "--checkpoint_path",   "/DATA1/Roxana/Hierarchy/Two_levels/hebi_foundation_best_new.pt",
        "--init_mode",         "pretrained",
        "--num_epochs",        "5",
        "--lr",                "1e-3",
        "--num_domains",       "10",
        "--comp_num_niches",   "10",
        "--spearman_threshold", "0.4",
        "--label_cols",        "response", "TRGStage",
        "--dense_top_n",       "1000",
    ]

    if not PER_SAMPLE:
        # original joint behaviour: one model over all samples of this type
        sys.argv = base_argv + ["--cancer_type", CANCER,
                                "--out_dir", f"./Xenium_CLINICAL_RAW/{CANCER}_joint"]
        results, all_grn_attn, trainer, gene_graphs_list = main(seed=1990)
    else:
        # per-sample: resolve the sample list once, then run main() per sample
        import scanpy as sc
        meta = sc.read_h5ad("/DATA2/Sally/data/xenium_data/adata_xenium.h5ad")
        samp_cancer = meta.obs.groupby("SampleID", observed=True)["CancerType"].first().to_dict()
        del meta

        all_files = sorted(os.path.basename(p)
                           for p in glob.glob(os.path.join(PT_DIR, "*.pt")))
        crc_files = [f for f in all_files
                     if samp_cancer.get(f.replace(".pt", "")) == CANCER]
        print(f"🎯 {CANCER}: {len(crc_files)} samples to run one-by-one")

        for k, f in enumerate(crc_files):
            sid = f.replace(".pt", "")
            one_dir = make_subset_dir(PT_DIR, [f], tag=f"{CANCER}_{sid}")
            out_dir = f"./Xenium_results_dense/pdac_persample_10_RAW_pretrained/{sid}"
            sys.argv = base_argv + ["--pt_dir", one_dir, "--out_dir", out_dir]

            results, all_grn_attn, trainer, gene_graphs_list = main(seed=1990)

            a = results[0]
            npz_path = os.path.join(out_dir, "sample_niches", f"{sid}.npz")
            if os.path.exists(npz_path):
                d = dict(np.load(npz_path, allow_pickle=True))
                if KEY_CELL_CLUSTERS in a.obs.columns:
                    d["z0_kmeans"] = a.obs[KEY_CELL_CLUSTERS].astype(int).values
                    np.savez(npz_path, **d)
                    print(f"✅ {sid}: z0_kmeans saved ({len(d['z0_kmeans'])} cells)")
                else:
                    print(f"⚠️ {sid}: no '{KEY_CELL_CLUSTERS}' column — cols: {list(a.obs.columns)}")
       