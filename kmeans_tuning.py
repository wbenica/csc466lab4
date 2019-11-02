import constants as c
from kmeans import kmeans, select_centroids_rand
from utils import *

DSS_DK = 'dsse/dk'


def hyper_tune_k(df, fn):
    table = pd.DataFrame(columns=[SSE, DSS_DK])
    best_k = (df.shape[0], float('-inf'))
    for k in range(1, df.shape[0]):
        clusters, centroids = kmeans(df, k, 1)
        measures = evaluate_clusters(clusters, centroids)
        table = table.append(pd.DataFrame([[measures[SSE].sum(), 0]], index=[k], columns=[SSE, DSS_DK]))
        if len(table) > 1:
            dss_dk = table.loc[k, SSE] - table.loc[k - 1, SSE]
            table.loc[k - 1, DSS_DK] = dss_dk
            if dss_dk > -1:
                print_and_plot(fn, table)
                return k - 2
            elif dss_dk + 1 > best_k[1]:
                best_k = (k - 2, dss_dk + 1)
    print_and_plot(fn, table)
    return best_k[0]


def hyper_tune_t(df, k):
    table = pd.DataFrame(columns=[SSE])
    for t in [(x + 1) for x in range(20)]:
        clusters, centroids = kmeans(df, k, t)
        measures = evaluate_clusters(clusters, centroids)
        curr_sse = pd.Series([measures[SSE].sum()], index=[SSE], name=t)
        table = table.append(pd.DataFrame([curr_sse]))
    min_sse = table[SSE].min()
    best = table[table[SSE] == min_sse]
    max_t = best.index.values.max()
    return max_t


def kmeans_hyper_tuning(fn):
    df, class_id = parse_csv(fn)
    k = hyper_tune_k(df, fn)
    t = hyper_tune_t(df, k)
    return k, t


def kmeans_k_t_selection():
    res = pd.DataFrame(columns=['k', 't'], index=c.ALL)
    for fn in c.ALL:
        k, t = kmeans_hyper_tuning(fn)
        res.at[fn, 'k'] = k
        res.at[fn, 't'] = t
    return res


def print_and_plot(fn, table):
    print(fn)
    print(table)
    print()
    fig, ax = plt.subplots()
    ax.plot(table.index.values, table.iloc[:, 0])
    sfn = fn.split('/')[-1].split('.')[0]
    ax.title.set_text(f'SSE vs k {sfn}')
    plt.savefig(f'./graphs/k_vs_sse/{sfn}')
    plt.show()


if __name__ == '__main__':
    print(kmeans_k_t_selection())


def kmeans_dist_and_centroid_selection():
    best_k_t = kmeans_k_t_selection()
    tests = ['rand/norm', 'rand/raw', 'smart/norm', 'smart/raw']
    for file in c.ALL:
        results = [pd.DataFrame()] * 4
        clusters = [pd.DataFrame()] * 4
        centroids = [pd.DataFrame()] * 4
        print(file)
        df, class_id = parse_csv(file)
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

        if clusters[0][0].shape[1] == 2:
            fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
            df_cent = np.array([df.mean().values])
            ax[0, 0].scatter(df.iloc[:, 0], df.iloc[:, 1])
            ax[0, 0].scatter(df_cent[0, 0], df_cent[0, 1], c='black')
            ax[0, 0].title.set_text('Dataset')
            ax[1, 0].axis('off')
            sfn = file.split('/')[-1].split('.')[0]
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
            title = f'./graphs/dist_cent_methods/{sfn}.png'
            plt.savefig(title)
            plt.show()
