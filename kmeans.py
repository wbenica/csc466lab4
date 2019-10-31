import random
import sys
import math
from typing import Union, List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import constants as c
from parse_data import parse_csv


# TODO: ACCIDENTS_3 has a comma at the end of every row, which messes things up for that dataset
def get_euclidean_distances_normalized(mx_one: Union[pd.DataFrame, np.ndarray],
                                       mx_two: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
    """normalizes all values to range 0.0-1.0 then calculates the sum of the manhattan distance between each row in
    mx_one and all the rows in mx_two
    :param mx_one: a DataFrame or np.ndarray of the data points whose distances you want to know
    :param mx_two: a DataFrame or np.ndarray of the data points from which you are calculating distance
    :return: an ndarray table of the distances between data points in mx_one and mx_2"""
    if mx_two is None:
        mx_two = mx_one
    if isinstance(mx_one, pd.DataFrame):
        mx_one: np.ndarray = mx_one.values
    if isinstance(mx_two, pd.DataFrame):
        mx_two: np.ndarray = mx_two.values
    one_ptp = mx_one.ptp(axis=0)
    one_min = mx_one.min(axis=0)
    mx_one = (mx_one - one_min) / one_ptp
    mx_two = (mx_two - one_min) / one_ptp
    mx_one_sq = np.square(mx_one).sum(axis=1)[:, np.newaxis]
    mx_two_sq = np.square(mx_two).sum(axis=1)
    mtx_prod = mx_one.dot(mx_two.transpose())
    return mx_one_sq + mx_two_sq - 2 * mtx_prod


def get_euclidean_distances(mx_one: Union[pd.DataFrame, np.ndarray],
                                       mx_two: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
    """calculates the sum of the manhattan distance between each row in mx_one and all
    the rows in mx_two
    :param mx_one: a DataFrame or np.ndarray of the data points whose distances you want to know
    :param mx_two: a DataFrame or np.ndarray of the data points from which you are calculating distance
    :return: an ndarray table of the distances between data points in mx_one and mx_2"""
    if mx_two is None:
        mx_two = mx_one
    if isinstance(mx_one, pd.DataFrame):
        mx_one: np.ndarray = mx_one.values
    if isinstance(mx_two, pd.DataFrame):
        mx_two: np.ndarray = mx_two.values
    mx_one_sq = np.square(mx_one).sum(axis=1)[:, np.newaxis]
    mx_two_sq = np.square(mx_two).sum(axis=1)
    mtx_prod = mx_one.dot(mx_two.transpose())
    return mx_one_sq + mx_two_sq - 2 * mtx_prod


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
    avg_dist = cluster.mean().sum(axis=0)
    return math.sqrt(avg_dist)


def get_sse(cluster: pd.DataFrame, centroid: np.ndarray) -> float:
    variance = cluster - centroid
    var_sq = np.square(variance).sum()
    return var_sq.sum()


def plot_clusters(clusters: List[pd.DataFrame], centroids: np.ndarray) -> None:
    """displays a scatterplot of clusters and their centroids
    :param clusters: a list of k DataFrames
    :param centroids: a 2D numpy array of the centroids of the clusters
    """
    fig, ax = plt.subplots()
    for cluster, centroid in zip(clusters, centroids):
        ax.scatter(cluster[0], cluster[1])
        ax.scatter(centroid[0], centroid[1], c='black')
    ax.grid(True)
    fig.tight_layout()

    plt.show()

def test():
    df = parse_csv(c.MANY_CLUSTERS)
    k = 5
    threshold = 0.1
    clusters, centroids = kmeans(df, k, threshold, get_dist=get_euclidean_distances_normalized)
    if 2 <= clusters[0].shape[1] <= 3:
        plot_clusters(clusters, centroids)
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
if __name__ == "__main__":
    test()
