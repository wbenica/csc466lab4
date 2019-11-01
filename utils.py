import math
from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

SSE = 'SSE'
PTS = 'Num Points'
AVG = 'Avg Dist'
MIN = 'Min Dist'
MAX = 'Max Dist'


def get_euclidean_distances(mx_one: Union[pd.DataFrame, np.ndarray],
                                       mx_two: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
    """calculates the manhattan distance between each row in mx_one and each of the rows in mx_two
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
    return np.sqrt(mx_one_sq + mx_two_sq - 2 * mtx_prod)


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
    return np.sqrt(mx_one_sq + mx_two_sq - 2 * mtx_prod)


def plot_clusters(clusters: List[pd.DataFrame], centroids: np.ndarray, title: str) -> None:
    """displays a scatterplot of clusters and their centroids
    :param clusters: a list of k DataFrames
    :param centroids: a 2D numpy array of the centroids of the clusters
    """
    if len(clusters) > 0 and 2 <= clusters[0].shape[1] <=3:
        ax: plt.Subplot = None
        if clusters[0].shape[1] == 2:
            fig, ax = plt.subplots()
            for cluster, centroid in zip(clusters, centroids):
                ax.scatter(cluster[0], cluster[1])
                if centroids is not None:
                    ax.scatter(centroid[0], centroid[1], c='black')
            ax.grid(True)
            fig.tight_layout()
        elif clusters[0].shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for cluster, centroid in zip(clusters, centroids):
                ax.scatter(cluster[0], cluster[1], cluster[2])
                if centroids is not None:
                    ax.scatter(centroid[0], centroid[1], centroid[2])
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


def parse_csv(fn: str) -> pd.DataFrame:
    with open(fn, 'r') as f:
        h = f.readline().split(',')
    header = to_header(h)
    if header[0] == 'row_id':
        df = pd.read_csv(fn, names=header[1:], index_col=0,skiprows=1)
    else:
        df = pd.read_csv(fn, names=header, skiprows=1)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(axis='columns')
    return df


def to_header(hrow):
    res = []
    if int(hrow[0]) == 0:
        res.append('row_id')
        res += list(range(0,len(hrow)-1))
    else:
        res = list(range(0,len(hrow)))
    return tuple(res)


def get_max_dist(cluster, centroid):
    cluster = np.absolute(cluster - centroid)
    max_dist = np.max(cluster.sum(axis=0))
    return math.sqrt(max_dist)


def get_min_dist(cluster, centroid):
    cluster = np.absolute(cluster - centroid)
    min_dist = np.min(cluster.sum(axis=0))
    return math.sqrt(min_dist)


def get_avg_dist(cluster, centroid):
    cluster = np.absolute(cluster - centroid)
    if len(cluster) > 0:
        avg_dist = cluster.mean().sum(axis=0)
        return math.sqrt(avg_dist)
    else:
        return 0


def get_sse(cluster: pd.DataFrame, centroid: np.ndarray) -> float:
    variance = cluster - centroid
    var_sq = np.square(variance).sum()
    return var_sq.sum()


def evaluate_clusters(clusters, centroids, verbose=False):
    results = pd.DataFrame(columns=[MAX, MIN, AVG, SSE, PTS])
    if centroids is None:
        centroids = clusters.mean()
    for i, clusters in enumerate(clusters):
        clust_vals = clusters.values if isinstance(clusters, pd.DataFrame) else clusters
        centroids = [centroid.value if isinstance(centroid, pd.DataFrame) else centroid for centroid in centroids]
        max = get_max_dist(clust_vals, centroids[i])
        min = get_min_dist(clust_vals, centroids[i])
        avg = get_avg_dist(clust_vals, centroids[i])
        num_points = len(clust_vals)
        sse = get_sse(clust_vals, centroids[i])
        data = pd.Series([max, min, avg, num_points, sse], name=str(i + 1), index=[MAX, MIN, AVG, PTS, SSE])
        results = results.append(data)
        if verbose:
            print()
            print(f'Cluster {i + 1}')
            print(f'Centroid: {centroids[i]}')
            print()
            print(clusters)
    return results


def drop_df(df1: pd.DataFrame, df2: pd.DataFrame):
    """removes rows from df1 that are also in df2"""
    return pd.concat([df1, df2, df2]).drop_duplicates(keep=False)
