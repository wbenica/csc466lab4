import random
import sys
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

import constants as c

# TODO: ACCIDENTS_3 has a comma at the end of every row, which messes things up for that dataset
from utils import get_euclidean_distances_normalized, get_euclidean_distances, plot_clusters, parse_csv


def shuffle(df: pd.DataFrame) -> np.ndarray:
    """shuffles df along axis 0 and returns it"""
    n = df.shape[0]
    indices = np.array(range(n))
    random.shuffle(indices)
    res = np.array([df.iloc[i] for i in indices])
    return res


# TODO: think about whether it's better to return a dataframe so that row_ids are passed along
def select_centroids_smart(df: pd.DataFrame, k: int, get_dist=get_euclidean_distances) -> np.ndarray:
    points = pd.DataFrame(df.mean(axis=0)).T
    i = 1
    while i < k:
        dists = get_dist(df, points).sum(axis=1)
        furthest = np.argmax(dists)
        next_point = pd.DataFrame(df.iloc[furthest]).T
        points = points.append(next_point)
        df = df.drop([df.index[furthest]])
        i += 1
    return points.values


def select_centroids_rand(df: pd.DataFrame, k: int) -> np.ndarray:
    """selects k random starting points for k-means clustering"""
    res = shuffle(df)
    return res[:k]


def check_centroid_change(old_centroids, new_centroids, threshold):
    change = abs((old_centroids - new_centroids).sum())
    print(f'change: {change}')
    return math.sqrt(change) < threshold


def check_num_reassignments(clusters, old_clusters):
    num_reassignments = 0
    if old_clusters is not None:
        for old_clust, clust in zip(old_clusters, clusters):
            old_clust = old_clust.index
            clust = clust.index
            for key in clust:
                if key not in old_clust:
                    num_reassignments += 1
        print(f'reassign: {num_reassignments}')
    else:
        return False
    return num_reassignments < 2


def check_sse_change(old_clusters, new_clusters, old_centroids, new_centroids):
    if old_clusters is None:
        return False
    old_sse = np.array([get_sse(old_clusters[i], old_centroids[i]) for i in range(len(old_clusters))]).sum()
    new_sse = np.array([get_sse(new_clusters[i], new_centroids[i]) for i in range(len(new_centroids))]).sum()
    change = new_sse - old_sse
    return change < 0 and change / old_sse < 0.005


def is_stopping_condition(old_clusters, new_clusters, old_centroids, new_centroids, threshold):
    num_reassigns = check_num_reassignments(new_clusters, old_clusters)
    change_centroids = check_centroid_change(old_centroids, new_centroids, threshold)
    sse_chng = check_sse_change(old_clusters, new_clusters, old_centroids, new_centroids)
    # sse_chng = False
    print('\nSTOP CHECK:')
    if num_reassigns:
        print('reass')
    if change_centroids:
        print('centrs')
    if sse_chng:
        print('sse')
    return change_centroids or num_reassigns or sse_chng


def kmeans(df: pd.DataFrame, k: int, threshold=None, select_centroids=select_centroids_smart,
           get_dist=get_euclidean_distances) -> Tuple[List[pd.DataFrame], np.ndarray]:
    centroids = select_centroids(df, k)
    old_clusters = None
    t = 0
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
        if t == 100:
            break
        # check stopping conditions
        if is_stopping_condition(old_clusters, clusters, centroids, new_centroids, threshold):
            print('stopped!')
            break
        centroids = new_centroids
        old_clusters = clusters
        t += 1
        # break
    return clusters, centroids


def get_max_dist(cluster, centroid):
    cluster = np.absolute(cluster.values - centroid)
    max_dist = np.max(cluster.sum(axis=0))
    return math.sqrt(max_dist)


def get_min_dist(cluster, centroid):
    cluster = np.absolute(cluster.values - centroid)
    min_dist = np.min(cluster.sum(axis=0))
    return math.sqrt(min_dist)


def get_avg_dist(cluster, centroid):
    cluster = np.absolute(cluster.values - centroid)
    if len(cluster) > 0:
        avg_dist = cluster.mean().sum(axis=0)
        return math.sqrt(avg_dist)
    else:
        return 0

def get_sse(cluster: pd.DataFrame, centroid: np.ndarray) -> float:
    variance = cluster - centroid
    var_sq = np.square(variance).sum()
    return var_sq.sum()


def test():
    fn = c.MANY_CLUSTERS
    df = parse_csv(fn)
    k = 4
    threshold = 0.1
    clusters, centroids = kmeans(df, k, threshold, get_dist=get_euclidean_distances)
    if 2 <= clusters[0].shape[1] <= 4:
        plot_clusters([df], np.array([df.mean().values]), f'kmeans {fn}')
        plot_clusters(clusters, centroids, f'kmeans clustered {fn}')
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
    if len(sys.argv) >= 4:
        threshold = sys.argv[3]
    else:
        threshold = None
    if len(sys.argv) >= 3:
        fn = sys.argv[1]
        k = sys.argv[2]
    else:
        raise TypeError(
            f'kmeans expected at least 2 arguments, got {len(sys.argv) - 1}')
    df = parse_csv(fn)
    kmeans(df, k, threshold)


if __name__ == "__main__":
    test()
