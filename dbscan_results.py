from constants import TOTALS
from dbscan import *


def dbscan_run_all():
    pd.set_option('expand_frame_repr', True)
    pd.set_option('max_rows', 100)
    pd.set_option('max_columns', 250)
    pd.set_option('display.width', 1000)
    np.set_printoptions(precision=3, floatmode='fixed')
    for fn in c.DB_TESTS:
        e, pts = c.e_pts[fn]
        df, class_id = parse_csv(fn)
        clusters, outliers = dbscan(df, e, pts)
        results = evaluate_clusters(clusters, None, verbose=False, outliers=outliers)
        totals = results.loc[:, [MAX, MIN, AVG, PTS, SSE]].sum()
        totals[NUM_DROPPED] = '-'
        totals[PCT_DROPPED] = '-'
        totals.name = TOTALS
        results = results.append(totals)
        sfn = strip_file_path(fn)
        print(f'\nSummary - {sfn}')
        print(results.round(3))
        centroids = np.array([cluster.mean() for cluster in clusters])
        for idx, (cluster, centroid) in enumerate(zip(clusters, centroids)):
            print(f'\nCluster {idx + 1}')
            print(f'Centroid: {centroid}')
            print(cluster)
        print('\nOutliers')
        print(outliers)
        if 2 <= clusters[0].shape[1] <= 3:
            plot_clusters(clusters, centroids, f'dbscan {sfn}')


if __name__ == '__main__':
    dbscan_run_all()
