"""Faithful(-ish) DEEP CLUSTER for the minority-examples split (Option A).

Reproduces the clustering half of the minority method from Reif & Schwartz (2023),
"Fighting Bias with Bias" -- the part our in-library ``minority_split`` only
*approximates* with kmeans/ward/deepcluster-lite. Their key move is DEEP CLUSTER
(Caron et al., 2018): rather than clustering fixed features, they train an encoder
to predict cluster pseudo-labels, then re-cluster its representation, which yields
the *label-diverse* clusters plain clustering can't.

This module owns all the heavy ML (torch + transformers + a training loop) and lives
entirely OUTSIDE the ``splytters`` package, which stays pure and gradient-free. It
produces a cluster assignment; routing that assignment into a train/test split is
delegated to the library's ``minority_route`` so the routing is byte-identical to
``minority_split`` (incl. footnote-10 ``minority_labels``).

Semi-faithful variant (default): the FIRST clustering uses precomputed frozen
sentence embeddings instead of a task-fine-tuned encoder, so only ONE fine-tune (the
pseudo-label pass) is needed. Pass ``task_finetune=True`` for the fully faithful two
fine-tune version.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def ward(emb: np.ndarray, k: int) -> np.ndarray:
    """Deterministic Ward clustering -- the paper's base clusterer."""
    k = min(k, len(emb))
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(emb)


def _finetune_and_extract(
    texts: list[str],
    pseudo_labels: np.ndarray,
    *,
    model_name: str = "roberta-base",
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_len: int = 64,
    device: str = "cpu",
    seed: int = 0,
    log=print,
) -> np.ndarray:
    """Fine-tune a FRESH ``model_name`` to predict ``pseudo_labels``, then return its
    encoder's [CLS] representation for every text. One DEEP CLUSTER iteration."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    k = int(pseudo_labels.max()) + 1
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=k
    ).to(device)

    enc = tok(texts, padding=True, truncation=True, max_length=max_len,
              return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(pseudo_labels, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        running = 0.0
        for step, (ids, mask, lab) in enumerate(dl, 1):
            opt.zero_grad()
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        labels=lab.to(device))
            out.loss.backward()
            opt.step()
            running += float(out.loss)
            if step % 50 == 0:
                log(f"  epoch {ep + 1}/{epochs} step {step}/{len(dl)} "
                    f"loss {running / step:.4f}")

    # Extract [CLS] from the base encoder (the reshaped representation).
    model.eval()
    base = model.base_model
    reps = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = tok(texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
            h = base(input_ids=b["input_ids"].to(device),
                     attention_mask=b["attention_mask"].to(device))
            reps.append(h.last_hidden_state[:, 0].cpu().numpy())
    return np.vstack(reps).astype(np.float32)


def deepcluster_labels(
    texts: list[str],
    initial_emb: np.ndarray,
    *,
    y: np.ndarray | None = None,
    n_clusters: int = 10,
    n_iters: int = 1,
    task_finetune: bool = False,
    log=print,
    **ft_kwargs,
) -> np.ndarray:
    """Return final cluster labels via DEEP CLUSTER.

    ``initial_emb`` seeds clustering #1 (frozen embeddings in the semi-faithful
    default). With ``task_finetune=True`` and ``y`` given, clustering #1 instead uses
    a task-fine-tuned encoder's [CLS] (fully faithful). Then, ``n_iters`` times:
    Ward-cluster -> fine-tune a fresh encoder on the pseudo-labels -> re-extract [CLS].
    """
    if task_finetune:
        if y is None:
            raise ValueError("task_finetune=True requires y (the task labels)")
        log("clustering #1 source: task-fine-tuned [CLS] (faithful)")
        rep = _finetune_and_extract(texts, np.asarray(y), log=log, **ft_kwargs)
    else:
        log("clustering #1 source: frozen embeddings (semi-faithful)")
        rep = np.asarray(initial_emb, dtype=np.float32)

    labels = ward(rep, n_clusters)
    for it in range(n_iters):
        log(f"deep-cluster iteration {it + 1}/{n_iters}: fine-tuning on pseudo-labels")
        rep = _finetune_and_extract(texts, labels, log=log, **ft_kwargs)
        labels = ward(rep, n_clusters)
    return labels
