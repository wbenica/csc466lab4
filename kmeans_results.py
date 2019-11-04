import numpy as np
import pandas as pd

import constants as c
from kmeans import kmeans, evaluate_clusters
from utils import parse_csv, plot_clusters, strip_file_path


def kmeans_run_all():
    pd.set_option('expand_frame_repr', True)
    pd.set_option('max_rows', 100)
    np.set_printoptions(precision=3, floatmode='fixed')
    for fn in c.ALL:
        k = c.ks[fn]
        t = 1
        df, class_id = parse_csv(fn)
        clusters, centroids = kmeans(df, k, t)
        results = evaluate_clusters(clusters, centroids, verbose=False)
        totals = results.sum()
        totals.name = c.TOTALS
        results = results.append(totals)
        sfn = strip_file_path(fn)
        print(f'\nSummary - {sfn}')
        print(results)
        for idx, (cluster, centroid) in enumerate(zip(clusters, centroids)):
            print(f'\nCluster {idx + 1}')
            print(f'Centroid: {centroid}')
            print(cluster)
        if 2 <= clusters[0].shape[1] <= 3:
            plot_clusters([df], np.array([df.mean().values]), f'kmeans {sfn}')
            plot_clusters(clusters, centroids, f'kmeans clustered {sfn}')


if __name__ == "__main__":
    print(kmeans_run_all())
