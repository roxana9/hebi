#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 14:57:55 2025

@author: tower4
"""

from continual import continual_pretrain
from encode_embeddings import encode_slide_to_z0_z1_z2
import scanpy as sc
import matplotlib.pyplot as plt

# --- 1) Continual pretraining on new .pt slides ---
checkpoint_path = "hebi_foundation_best2.pt"  # pretrained foundation model
new_data_dir = "/DATA1/Roxana/Hierarchy/processed_toy"  # unseen synthetic slides
out_ckpt = "hebi_foundation_continual.pt"

trainer = continual_pretrain(
    checkpoint_path=checkpoint_path,
    new_data_dir=new_data_dir,
    device="cuda:0",
    lr=1e-4,
    num_epochs=100,
    out_checkpoint_path=out_ckpt,
)

# --- 2) Pick a specific slide to encode (one of the .pt files in new_data_dir) ---
slide_path = f"{new_data_dir}/toy0.pt"

z0, z1, z2, smooth_labels = encode_slide_to_z0_z1_z2(trainer, slide_path)

# If you want numpy arrays:
z0_np = z0.cpu().numpy()
z1_np = z1.cpu().numpy()
z2_np = z2.cpu().numpy()
smooth_np = smooth_labels.numpy()
adata = sc.read_h5ad("/DATA1/Roxana/Hierarchy/Data/slide_0_0.h5ad")

adata.obs['smooth']=smooth_np
adata.obs['smooth']=adata.obs['smooth'].astype('category')

sc.pl.embedding(adata, basis="spatial", color="smooth",  show=False,title="smooth_emb",size=10)
plt.gca().invert_yaxis()  # Invert the y-axis to match the original plot style
plt.axis('off')
#plt.legend(loc="center right", fontsize=12, markerscale=1,bbox_to_anchor=(1.1, 0.5), borderaxespad=0., frameon=False)
#plt.savefig(f"SemanticST_{s}_{d}", dpi=600, bbox_inches='tight')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=20, random_state=1) 
embedding = pca.fit_transform(z0_np)
adata.obsm['z0emb'] = embedding
# sc.pp.neighbors(adata, use_rep='z0emb')
# sc.tl.umap(adata)
# sc.tl.leiden(adata, random_state=2025, resolution=1)
# adata.obs['leiden'] = adata.obs['leiden'].astype('category')
# # plt.rcParams["figure.figsize"] = (10, 7)


# Plot all clusters together

from sklearn.cluster import KMeans

k = 7

X = adata.obsm["z0emb"]

kmeans = KMeans(
    n_clusters=k,
    random_state=0,
    n_init="auto",
)
labels = kmeans.fit_predict(X)

# First store as strings
adata.obs["kmeans_z0"] = labels.astype(str)

# # Then convert that column to categorical (pandas)
adata.obs["kmeans_z0"] = adata.obs["kmeans_z0"].astype("category")
sc.pl.embedding(adata, basis="spatial", color="mclust_raw", show=False, title="pca",size=10)
plt.gca().invert_yaxis()  # Invert the y-axis to match the original plot style
plt.axis('off')
#plt.legend(loc="center right", fontsize=12, markerscale=1,bbox_to_anchor=(1.1, 0.5), borderaxespad=0., frameon=False)
#plt.savefig(f"SemanticST_{s}_{d}", dpi=600, bbox_inches='tight')
plt.show()
from sklearn.metrics import adjusted_rand_score

gt = adata.obs["domain"]

ari_z0 = adjusted_rand_score(gt, adata.obs["kmeans_z0"])
# ari_z1 = adjusted_rand_score(gt, adata.obs["kmeans_z1"])
# ari_z2 = adjusted_rand_score(gt, adata.obs["kmeans_z2"])

print(f"ARI(kmeans_z0 vs domain): {ari_z0:.4f}")
# print(f"ARI(kmeans_z1 vs domain): {ari_z1:.4f}")
# print(f"ARI(kmeans_z2 vs domain): {ari_z2:.4f}")



