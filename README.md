# HEBI

**A hierarchical graph foundation model for spatial transcriptomics.**

HEBI learns cell, prototype, and niche level representations of a tissue
section in a single forward pass, without cell type or niche annotations.
It is pretrained on 80 imaging-based sections spanning three commercial
platforms and 23 million cells, and transfers to unseen sections either
zero-shot or after a short continual adaptation.

<p align="center">
  <img src="docs/images/hebi_overview.png" alt="HEBI overview" width="800"/>
</p>

Niches are read directly from the model's attention rather than by
clustering an embedding afterwards, so niche labels are comparable across
replicate sections and across platforms without a separate batch
correction step.

---

## Installation

HEBI needs Python 3.11+, PyTorch and `torch_sparse`. Both are built per
CUDA version, so they are installed first, then the package.

The commands below use PyTorch 2.6.0 with CUDA 12.4, the versions HEBI was
developed with. For another CUDA version, or CPU only, change `cu124` in
both URLs (e.g. `cu118`, `cu126`, `cpu`); see
[pytorch.org](https://pytorch.org/get-started/locally/) and the
[PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### conda / mamba

`environment.yml` installs PyTorch and `torch_sparse` through pip, since
PyTorch is no longer published on its conda channel after 2.5.

```bash
git clone https://github.com/roxana9/hebi.git
cd hebi
mamba env create -f environment.yml
mamba activate hebi
pip install -e .
```

### pip

```bash
# 1. PyTorch, matching your CUDA version
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. torch_sparse, a prebuilt wheel matching that PyTorch + CUDA build
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 3. HEBI
git clone https://github.com/roxana9/hebi.git
cd hebi
pip install -e .
```

If `torch_sparse` starts compiling from source, the `torch-X+cuY` part of
the `-f` URL does not match the installed PyTorch. Check it with
`python -c "import torch; print(torch.__version__)"`.

Optional clustering backends:

```bash
pip install -e ".[leiden]"    # --cluster_method leiden
pip install -e ".[mclust]"    # --cluster_method mclust, also needs R + mclust
```

Check the install:

```bash
hebi-run --help
```

---

## Pretrained weights

<!-- TODO: upload checkpoints and fill in the download instructions.

Two checkpoints will be released:
  - trained on raw counts
  - trained on library-normalised, log1p data

Use the one matching how your sections were preprocessed: a section
encoded in a different space from the one the model was pretrained on
will give degraded results without any error.
-->

Pretrained weights are not yet released. Until then, `--init random`
trains from scratch on your own sections.

---

## Quick start

Three steps: preprocess sections, run inference, read the outputs.

### 1. Preprocess

Turns each `.h5ad` into a `.pt` section holding the expression matrix,
the spatial kNN graph, and the METIS partition.

```bash
hebi-preprocess \
  --h5ad_dir data/my_sections/ \
  --out_root processed/ \
  --vocab_path default_gene_vocab.json \
  --k 5
```

`--k` is the spatial neighbourhood size and should suit the cell density
of your data. The value used is recorded in each `.pt`.

Normalisation is detected, not assumed. `log1p` is applied when the
matrix is not already in log space; library-size normalisation is
separate and opt-in via `--normalise`. The detected status is printed and
stored, so a section's space is always recoverable.

### 2. Run

```bash
hebi-run \
  --checkpoint_path hebi_foundation.pt \
  --data_dir processed/ \
  --section_files section_1.pt section_2.pt \
  --h5ad_files data/my_sections/section_1.h5ad data/my_sections/section_2.h5ad \
  --vocab_path default_gene_vocab.json \
  --n_cell_clusters 16 \
  --n_composition_niches 12 \
  --output_dir results/
```

Or put the same settings in a YAML file:

```bash
hebi-run --config configs/starmap.yaml
```

### 3. Outputs

Written to `--output_dir`:

```
joint_cell_clusters_z0.png / .svg     all sections side by side
joint_composition_niches.png / .svg
<section>_plot/                       per-section figures
```

And in each section's `.h5ad` object:

| `adata.obs` | meaning |
| --- | --- |
| `prototypes` | Level-2 attention argmax, straight from the forward pass |
| `cell_clusters` | clustering of the cell-level `z0` |
| `composition_niches` | k-means on local prototype composition |

---

## Python API

```python
from hebi.inference.pipeline import run_hebi

sections, trainer, grn_attention = run_hebi(
    checkpoint_path="hebi_foundation.pt",
    data_dir="processed/",
    section_files=["section_1.pt", "section_2.pt"],
    h5ad_files=["section_1.h5ad", "section_2.h5ad"],
    vocab_path="default_gene_vocab.json",
    n_cell_clusters=16,
    n_composition_niches=12,
)
```

Or use the pieces directly:

```python
from hebi.inference.encode import load_hebi, encode_section

model = load_hebi("hebi_foundation.pt", "processed/", "default_gene_vocab.json")
out = encode_section(model, "processed/section_1.pt")
# out["z0"], out["z1"], out["z2"], out["prototypes"], out["niche_attention"]
```

---

## What else it does

**Zero-shot or continual.** `--mode zero_shot` encodes with frozen
weights; `--mode continual` adapts to the target sections first with the
same self-supervised objective.

**Gene regulatory networks.** `--grn` builds a heterogeneous
gene-interaction graph from curated priors and co-expression, and learns
an attention over it. Gives hub genes, edge salience, and niche-level
regulatory activity.

**Cell-cell communication.** `--grn_ccc` scores ligand-receptor
communication between niches and writes a chord diagram plus the full
interaction table.

**Supervised niche prediction.** A linear head on the cell-level
representation, trained by linear probing and optional fine-tuning, with
sections held out whole so reported performance reflects generalisation
to unseen sections.

---

## Pretraining

To pretrain from scratch on your own corpus:

```bash
hebi-pretrain --config configs/pretrain.yaml
```

The config carries the architecture, the optimisation settings, and the
weights of the seven self-supervised loss terms.

---

## Documentation

<!-- TODO: point at the docs site once it is published.
     Tutorials to add: preprocessing, STARmap walkthrough,
     niche prediction, GRN and cell-cell communication. -->

Tutorials are in preparation.

---

## Citation

Manuscript in preparation.

---

## Licence

MIT. See [LICENSE](LICENSE).
