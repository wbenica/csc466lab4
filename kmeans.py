import random
import sys

import numpy as np
import pandas as pd
from typing import Union

import constants as c
from parse_data import parse_csv


def get_manhattan_distances(mx_one: Union[pd.DataFrame, np.ndarray], mx_two: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
    """calculates the sum of the manhattan distance between each row in mx_one and all
    the rows in mx_two
    :param mx_one: a DataFrame or np.ndarray of the data points whose distances you want to know
    :param mx_two: a DataFrame or np.ndarray of the data points from which you are calculating distance
    :return: an ndarray table of the distances between data points in mx_one and mx_2"""
    if mx_two is None:
        mx_two = mx_one
    mx_one = mx_one.values
    if isinstance(mx_two, pd.DataFrame):
        mx_two = mx_two.values
    mx_one_sq = np.square(mx_one).sum(axis=1)[:, np.newaxis]
    mx_two_sq = np.square(mx_two).sum(axis=1)
    mtx_prod = mx_one.dot(mx_two.transpose())
    return np.sqrt(mx_one_sq + mx_two_sq - 2 * mtx_prod)


def get_euclidean_distances(df: pd.DataFrame):
    pass


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    """shuffles df along axis 0 and returns it"""
    n = df.shape[0]
    indices = np.array(range(n))
    random.shuffle(indices)
    res = pd.DataFrame([df.iloc[i] for i in indices])
    return res


# TODO: think about whether it's better to return a dataframe so that row_ids are passed along
def select_centroids_smart(df: pd.DataFrame, k: int, get_dist=get_manhattan_distances) -> np.ndarray:
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


def select_centroids_rand(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """selects k random starting points for k-means clustering"""
    res = shuffle(df)
    return res[:k]


def kmeans(df: pd.DataFrame, k: int, threshold=None, select_centroids=select_centroids_smart,
           get_dist=get_manhattan_distances) -> None:
    centroids = select_centroids(df, k)
    while True:
        # get dist to centroids
        dists = get_dist(df, centroids)
        cluster_rankings = np.argsort(np.argsort(dists))

        clusters = []
        for i in range(k):
            mask = cluster_rankings[:, i] == 0
            clusters.append(df[mask])
        # check stopping conditions
        break
    print()



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


def test():
    df = parse_csv(c.PLANETS)
    k = 3
    threshold = None
    kmeans(df, k, threshold)


# TODO: add command line options for centroid select and get dist, using getopts?
if __name__ == "__main__":
    test()
