from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import dendrogram

from tqdm import tqdm


def preprocess_column(col):
    col = col.fillna("").astype(str)
    return " ".join(col)


def fit_model(columns):
    start = time()

    print("Preprocessing columns")
    columns = map(preprocess_column, columns)

    print("Extracting features")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tqdm(columns)).toarray()

    print("Creating distance matrix")
    X = np.dot(X, X.T) / np.linalg.norm(X, axis=1) / np.linalg.norm(X, axis=1)[:, None]
    X = 1 - X

    print("Clustering")
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="single",
        distance_threshold=0.35,
    )
    clusters = model.fit_predict(X)

    end = time()
    print(f"Time elapsed: {end - start:.2f}s")

    return clusters.tolist()
