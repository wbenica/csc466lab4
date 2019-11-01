import constants as c
from kmeans import kmeans, select_centroids_rand
from kmeans_hyperparams import kmeans_k_t_selection
from utils import *


def kmeans_dist_and_centroid_selection():
    best_k_t = kmeans_k_t_selection()
    for file in c.TWO_DIM:
        results = [pd.DataFrame()] * 4
        clusters = [pd.DataFrame()] * 4
        centroids = [pd.DataFrame()] * 4
        tests = ['rand/norm', 'rand/raw', 'smart/norm', 'smart/raw']
        print(file)
        df = parse_csv(file)
        print('CENTROID SELECTION: random')
        print('DISTANCES: normalized')
        clusters[0], centroids[0] = kmeans(df, best_k_t.loc[file, 'k'], best_k_t.loc[file, 't'],
                                           select_centroids=select_centroids_rand,
                                           get_dist=get_euclidean_distances_normalized)
        results[0] = evaluate_clusters(clusters[0], centroids[0], verbose=True)
        totals: pd.Series = results[0].sum()
        totals.name = 'total'
        results[0] = results[0].append(totals)
        print(results[0])
        print('\n\n')

        print('CENTROID SELECTION: random')
        print('DISTANCES: raw')
        clusters[1], centroids[1] = kmeans(df, best_k_t.loc[file, 'k'], best_k_t.loc[file, 't'],
                                           select_centroids=select_centroids_rand)
        results[1] = evaluate_clusters(clusters[1], centroids[1], verbose=True)
        totals: pd.Series = results[1].sum()
        totals.name = 'total'
        results[1] = results[1].append(totals)
        print(results[1])
        print('\n\n')

        print('CENTROID SELECTION: smart')
        print('DISTANCES: normalized')
        clusters[2], centroids[2] = kmeans(df, best_k_t.loc[file, 'k'], best_k_t.loc[file, 't'],
                                           get_dist=get_euclidean_distances_normalized)
        results[2] = evaluate_clusters(clusters[2], centroids[2], verbose=True)
        totals: pd.Series = results[2].sum()
        totals.name = 'total'
        results[2] = results[2].append(totals)
        print(results[2])
        print('\n\n')

        print('CENTROID SELECTION: smart')
        print('DISTANCES: raw')
        clusters[3], centroids[3] = kmeans(df, best_k_t.loc[file, 'k'], best_k_t.loc[file, 't'])
        results[3] = evaluate_clusters(clusters[3], centroids[3], verbose=True)
        totals: pd.Series = results[3].sum()
        totals.name = 'total'
        results[3] = results[3].append(totals)
        print(results[3])
        print('\n\n')

        print('\n\n')

        fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
        df_cent = np.array([df.mean().values])
        ax[0, 0].scatter(df.iloc[:, 0], df.iloc[:, 1])
        ax[0, 0].scatter(df_cent[0, 0], df_cent[0, 1], c='black')
        ax[0, 0].title.set_text('Dataset')
        ax[1, 0].axis('off')
        for i in range(4):
            row = i // 2
            col = i % 2 + 1
            for cluster, centroid in zip(clusters[i], centroids[i]):
                ax[row, col].scatter(cluster[0], cluster[1])
                if centroid is not None:
                    ax[row, col].scatter(centroid[0], centroid[1], c='black')
                ax[row, col].set_title(tests[i])
                ax[row, col].grid(True)
        plt.tight_layout()
        plt.show()


def test_dbscan():
    pass


if __name__ == "__main__":
    kmeans_dist_and_centroid_selection()
