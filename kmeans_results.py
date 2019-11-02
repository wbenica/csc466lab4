import numpy as np

import constants as c
from kmeans import kmeans, evaluate_clusters
from utils import parse_csv, plot_clusters


def kmeans_run_all():
    for fn in c.ALL:
        k = c.ks[fn]
        t = 1
        df, class_id = parse_csv(fn)
        clusters, centroids = kmeans(df, k, t)
        results = evaluate_clusters(clusters, centroids, verbose=True)
        totals = results.sum()
        totals.name = 'totals'
        results = results.append(totals)
        print(f'\n{fn} Summary')
        print(results)
        if 2 <= clusters[0].shape[1] <= 3:
            sfn = fn.split('/')[-1].split('.')[0]
            plot_clusters([df], np.array([df.mean().values]), f'kmeans {sfn}')
            plot_clusters(clusters, centroids, f'kmeans clustered {sfn}')


if __name__ == "__main__":
    print(kmeans_run_all())
