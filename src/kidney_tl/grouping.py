"""Recover scans so that no scan is split between training and testing.

The dataset has no patient identifiers, but within each class the file numbers
follow slice order: an image's most similar image is almost always the next
file number (median correlation 0.999 between neighbours versus about 0.4 for
random pairs). A drop in similarity between consecutive numbers marks the start
of a new scan. Some scans are also stored twice under different numbers
(byte-identical or near-identical slices), so segments that share such images
are merged into one group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def scan_segments(df: pd.DataFrame, thumbs: np.ndarray, break_r: float) -> np.ndarray:
    """Segment id per image: consecutive slices of one class stay together until similarity drops."""
    segment = np.empty(len(df), dtype=np.int64)
    next_id = 0
    for _, sub in df.groupby("label", sort=True):
        order = sub.index.to_numpy()[np.argsort(sub["slice"].to_numpy(), kind="stable")]
        r = np.einsum("ij,ij->i", thumbs[order[:-1]], thumbs[order[1:]])
        local = np.concatenate([[0], np.cumsum(r < break_r)])
        segment[order] = local + next_id
        next_id += local.max() + 1
    return segment


def duplicate_links(df: pd.DataFrame, md5: np.ndarray, thumbs: np.ndarray, segment: np.ndarray,
                    duplicate_r: float, block: int = 1024) -> np.ndarray:
    """Pairs of segments (same class) that contain a byte-identical or near-identical image."""
    labels = df["label"].to_numpy()
    links = set()
    by_hash = pd.Series(segment).groupby(md5).unique()
    for segs in by_hash[by_hash.map(len) > 1]:
        links |= {(segs[0], s) for s in segs[1:]}
    for start in range(0, len(df), block):
        sim = thumbs[start:start + block] @ thumbs.T
        rows, cols = np.nonzero(sim >= duplicate_r)
        rows += start
        keep = (segment[rows] != segment[cols]) & (labels[rows] == labels[cols])
        links |= set(zip(segment[rows[keep]].tolist(), segment[cols[keep]].tolist(), strict=True))
    return np.array(sorted(links), dtype=np.int64).reshape(-1, 2)


def assign_groups(df: pd.DataFrame, md5: np.ndarray, thumbs: np.ndarray, break_r: float,
                  duplicate_r: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Group id per image (scan segments merged across duplicates) and a per-class summary."""
    segment = scan_segments(df, thumbs, break_r)
    links = duplicate_links(df, md5, thumbs, segment, duplicate_r)
    n = segment.max() + 1
    graph = coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(n, n)) if len(links) else \
        coo_matrix((n, n))
    _, merged = connected_components(graph, directed=False)
    group = merged[segment]
    info = []
    for cls, sub in df.groupby("cls", sort=True):
        idx = sub.index.to_numpy()
        sizes = np.bincount(np.unique(group[idx], return_inverse=True)[1])
        info.append({"class": cls, "images": len(idx), "scan_segments": int(np.unique(segment[idx]).size),
                     "groups": len(sizes), "median_group_size": float(np.median(sizes)),
                     "largest_group": int(sizes.max())})
    return group, pd.DataFrame(info)


def max_similarity(query: np.ndarray, reference: np.ndarray, thumbs: np.ndarray, block: int = 1024) -> np.ndarray:
    """For each query image, its highest correlation with any reference image."""
    out = np.empty(len(query), dtype=np.float32)
    ref = thumbs[reference]
    for start in range(0, len(query), block):
        out[start:start + block] = (thumbs[query[start:start + block]] @ ref.T).max(axis=1)
    return out
