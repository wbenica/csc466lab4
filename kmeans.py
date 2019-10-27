import random
import sys

import numpy as np
import pandas as pd

import constants as c
from parse_data import parse_csv


def get_manhattan_distances(mx_one, mx_two=None):
    if mx_two is None:
        mx_two = mx_one
    mx_one = mx_one.values
    mx_two = mx_two.values
    mx_one_sq = np.square(mx_one).sum(axis=1)[:, np.newaxis]
    mx_two_sq = np.square(mx_two).sum(axis=1)
    mtx_prod = mx_one.dot(mx_two.transpose())
    return np.sqrt(mx_one_sq + mx_two_sq - 2 * mtx_prod)


def get_euclidean_distances(df: pd.DataFrame):
    pass


def shuffle(df: pd.DataFrame):
    n = df.shape[0]
    indices = np.array(range(n))
    random.shuffle(indices)
    res = pd.DataFrame([df.iloc[i] for i in indices])
    return res


def select_centroids_smart(df: pd.DataFrame, k: int) -> pd.DataFrame:
    points = pd.DataFrame(df.mean(axis=0)).T
    i = 1
    while i < k:
        dists = get_manhattan_distances(df, points).sum(axis=1)
        furthest = np.argmax(dists)
        next_point = pd.DataFrame(df.iloc[furthest]).T
        points = points.append(next_point)
        df = df.drop([df.index[furthest]])
        i += 1
    return points


def select_centroids_rand(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """selects k random starting points for k-means clustering"""
    res = shuffle(df)
    return res[:k]


def kmeans(df: pd.DataFrame, k: int) -> None:
    pass


def main():
    if len(sys.argv) == 3:
        fn = sys.argv[1]
        k = sys.argv[2]
    else:
        raise TypeError(f'kmeans expected at least 2 arguments, got {len(sys.argv) - 1}')
    df = parse_csv(fn)
    kmeans(df, k)


def test():
    df = parse_csv(c.PLANETS)
    k = 3
    centroids = select_centroids_smart(df, k)
    print(centroids)


if __name__ == "__main__":
    test()
