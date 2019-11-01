import matplotlib.pyplot as plt
import pandas as pd

import constants as c
from kmeans import kmeans
from utils import SSE, evaluate_clusters, parse_csv


def hyper_tune_k(df):
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
                fig, ax = plt.subplots()
                ax.plot(table.index.values, table.iloc[:, 0])
                plt.show()
                return k - 2
            elif dss_dk + 1 > best_k[1]:
                best_k = (k - 2, dss_dk + 1)
    fig, ax = plt.subplots()
    ax.plot(table.index.values, table.iloc[:, 0])
    plt.show()
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
    df = parse_csv(fn)
    k = hyper_tune_k(df)
    t = hyper_tune_t(df, k)
    return k, t


def kmeans_k_t_selection():
    res = pd.DataFrame(columns=['k', 't'], index=c.ALL)
    for fn in c.ALL:
        k, t = kmeans_hyper_tuning(fn)
        res.at[fn, 'k'] = k
        res.at[fn, 't'] = t
    return res


DSS_DK = 'dsse/dk'
