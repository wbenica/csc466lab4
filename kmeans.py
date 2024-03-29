import math
import random
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

import constants as c
from constants import KMEANS_CENTROID_THRESHOLD
from utils import drop_df, strip_file_path

from utils import *


def shuffle(df: pd.DataFrame) -> np.ndarray:
    """shuffles df along axis 0 and returns it"""
    n = df.shape[0]
    indices = np.array(range(n))
    random.shuffle(indices)
    res = np.array([df.iloc[i] for i in indices])
    return res


def select_centroids_smart(df: pd.DataFrame, k: int, get_dist=get_euclidean_distances) -> np.ndarray:
    points = pd.DataFrame(df.mean(axis=0)).T
    i = 1
    while i < k:
        dists = get_dist(df, points).sum(axis=1)
        furthest = np.argmax(dists)
        next_point = pd.DataFrame(df.iloc[furthest]).T
        points = points.append(next_point)
        df = drop_df(df, df.iloc[furthest])
        i += 1
    return points.values


def select_centroids_rand(df: pd.DataFrame, k: int) -> np.ndarray:
    """selects k random starting points for k-means clustering"""
    res = shuffle(df)
    return res[:k]


def check_centroid_change(old_centroids, new_centroids):
    if len(new_centroids) == 0:
        return False
    change = abs((old_centroids - new_centroids).sum())
    return math.sqrt(change) < KMEANS_CENTROID_THRESHOLD


def check_num_reassignments(clusters, old_clusters):
    num_reassignments = 0
    if old_clusters is not None:
        for old_clust, clust in zip(old_clusters, clusters):
            old_clust = old_clust.index
            clust = clust.index
            for key in clust:
                if key not in old_clust:
                    num_reassignments += 1
    else:
        return False
    return num_reassignments < 2


def check_sse_change(old_clusters, new_clusters, old_centroids, new_centroids, threshold):
    if old_clusters is None:
        return False
    old_sse = np.array([get_sse(old_clusters[i], old_centroids[i]) for i in range(len(old_clusters))]).sum()
    new_sse = np.array([get_sse(new_clusters[i], new_centroids[i]) for i in range(len(new_centroids))]).sum()
    change = new_sse - old_sse
    return abs(change) / old_sse < threshold


def is_stopping_condition(old_clusters, new_clusters, old_centroids, new_centroids, threshold):
    num_reassigns = check_num_reassignments(new_clusters, old_clusters)
    change_centroids = check_centroid_change(old_centroids, new_centroids)
    sse_chng = check_sse_change(old_clusters, new_clusters, old_centroids, new_centroids, threshold)
    return change_centroids or num_reassigns or sse_chng


def kmeans(df: pd.DataFrame, k: int, threshold=None, select_centroids=select_centroids_smart,
           get_dist=get_euclidean_distances) -> Tuple[List[pd.DataFrame], np.ndarray]:
    centroids = select_centroids(df, k)
    old_clusters = None
    while True:
        # get distances to centroids
        dists = get_dist(df, centroids)
        cluster_rankings = np.argsort(np.argsort(dists))
        # make clusters
        clusters: List[pd.DataFrame] = []
        for i in range(k):
            mask = cluster_rankings[:, i] == 0
            clusters.append(df[mask])
        new_centroids = np.array([cluster.mean() for cluster in clusters])
        # check stopping conditions
        if is_stopping_condition(old_clusters, clusters, centroids, new_centroids, threshold):
            break
        centroids = new_centroids
        old_clusters = clusters
        # break
    return clusters, centroids


def test():
    fn = c.PLANETS
    df, class_id = parse_csv(fn)
    k = c.ks[fn]
    threshold = c.KMEANS_SSE_THRESHOLD
    clusters, centroids = kmeans(df, k, threshold)
    sfn = strip_file_path(fn)
    if 2 <= clusters[0].shape[1] <= 4:
        plot_clusters([df], np.array([df.mean().values]), f'kmeans {sfn}')
        plot_clusters(clusters, centroids, f'kmeans clustered {sfn}')
    for i, cluster in enumerate(clusters):
        print()
        print(f'Cluster {i + 1}')
        print(f'Centroid: {centroids[i]}')
        print(f'Max Dist: {get_max_dist(cluster, centroids[i])}')
        print(f'Min Dist: {get_min_dist(cluster, centroids[i])}')
        print(f'Avg Dist: {get_avg_dist(cluster, centroids[i])}')
        print(f'Num. Points: {len(cluster)}')
        print(f'SSE: {get_sse(cluster, centroids[i])}')
        print()
        print(cluster)


# TODO: add command line options for centroid select and get dist, using getopts?
def main():
    np.set_printoptions(precision=3, floatmode='fixed')
    pd.options.display.float_format = '{:.3f}'.format
    if len(sys.argv) >= 4:
        threshold = float(sys.argv[3])
    else:
        threshold = c.KMEANS_SSE_THRESHOLD
    if len(sys.argv) >= 3:
        fn = sys.argv[1]
        k = int(sys.argv[2])
    else:
        raise TypeError(
            f'kmeans expected at least 2 arguments, got {len(sys.argv) - 1}')
    df, class_id = parse_csv(fn)
    clusters, centroids = kmeans(df, k, threshold)
    results = evaluate_clusters(clusters, centroids, verbose=False)
    if class_id is not None:
        accuracy = evaluate_classes(clusters, class_id)
    totals = results.sum()
    totals.name = 'totals'
    results = results.append(totals)
    sfn = strip_file_path(fn)
    print('\nSummary')
    print(results)
    if 2 <= clusters[0].shape[1] <= 3:
        plot_clusters([df], np.array([df.mean().values]), f'kmeans {sfn}')
        plot_clusters(clusters, centroids, f'kmeans clustered {sfn}')


if __name__ == "__main__":
    main()
