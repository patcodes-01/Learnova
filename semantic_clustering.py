from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os

# Load semantic encoder once (global)

MODEL_PATH = os.path.join("models", "pyq_semantic_encoder")
_model = SentenceTransformer(MODEL_PATH)


def embed_phrases(phrases: List[str]) -> np.ndarray:
    """
    Takes a list of phrases (RAKE output) and returns encoder embeddings.
    """
    if not phrases:
        return np.empty((0, 384))  

   
    clean_phrases = [str(p).strip() for p in phrases if str(p).strip()]
    if not clean_phrases:
        return np.empty((0, 384))

    embeddings = _model.encode(
        clean_phrases,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings


def cluster_topics_semantic(
    phrases: List[str],
    distance_threshold: float = 1.2,
    min_cluster_size: int = 2,
) -> Dict[int, List[str]]:
    """
    Use encoder embeddings + Agglomerative Clustering to group semantically
    similar topic phrases (RAKE output).

    Returns:
        {cluster_id: [phrase1, phrase2, ...]}
    """
    # 1) Deduplicate phrases while preserving order
    seen = set()
    unique_phrases = []
    for p in phrases:
        p = str(p).strip()
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            unique_phrases.append(p)

    if not unique_phrases:
        return {}

    # 2) Get embeddings
    emb = embed_phrases(unique_phrases)
    if emb.shape[0] == 0:
        return {}

    # 3) Agglomerative clustering on embeddings
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="euclidean",      
        linkage="ward",
    )
    labels = clustering.fit_predict(emb)

    # 4) Build {cluster_id: [phrases]}
    clusters: Dict[int, List[str]] = {}
    for phrase, label in zip(unique_phrases, labels):
        clusters.setdefault(int(label), []).append(phrase)

    # 5) Filter out tiny clusters
    clusters = {
        cid: plist
        for cid, plist in clusters.items()
        if len(plist) >= min_cluster_size
    }

    return clusters
