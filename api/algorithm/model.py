from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from tqdm import tqdm


def preprocess_column(col, nospace=False):
    col = col.fillna("").astype(str).str.lower()
    if nospace:
        col = col.str.replace(" ", "_")
    return " ".join(col)


def compute_similarity_matrix(X, alpha=0.5):
    batch_size = 50
    n = X.shape[0]

    sizes = np.sqrt(X.power(2).sum(axis=1) + 1)
    weight = np.exp(-alpha * np.abs(sizes[None, :] - sizes[:, None]))
    weight /= np.outer(sizes, sizes)
    weight[np.isnan(weight) | np.isinf(weight)] = 1
    weight *= 2

    sim = np.zeros((n, n))
    for i in tqdm(range(0, n, batch_size)):
        j = min(i + batch_size, n)
        sim[i:j] = cosine_similarity(X[i:j], X)
        sim[i:j] *= weight[i:j]

    sim[np.isnan(sim)] = 0
    return sim


def fit_model(columns):
    start = time()

    print("Preprocessing columns")
    num_columns = len(columns)
    columns_space = map(preprocess_column, columns)
    columns_nospace = map(lambda x: preprocess_column(x, nospace=True), columns)

    print("Extracting features")
    vectorizer = TfidfVectorizer(token_pattern=r"\S+")

    X_space = vectorizer.fit_transform(tqdm(columns_space, total=num_columns))
    sim_space = compute_similarity_matrix(X_space, alpha=0.3)

    X_nospace = vectorizer.fit_transform(tqdm(columns_nospace, total=num_columns))
    sim_nospace = compute_similarity_matrix(X_nospace, alpha=0.6)

    sim = (sim_space + sim_nospace) / 2
    X = 1 - sim

    print("Clustering")
    model = HDBSCAN(min_cluster_size=2, metric="precomputed")
    clusters = model.fit_predict(X)

    end = time()
    print(f"Time elapsed: {end - start:.2f}s")

    return clusters.tolist()
