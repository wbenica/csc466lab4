from dbscan import *


def dbscan_hypertuning(fn: str):
    df, class_id = parse_csv(fn)
    # best: k, pts, SSE
    best = (0, 0, 0)
    for e in range(1, 50, 2):
        for pts in range(1, 10):
            clusters = dbscan(df, e, pts)
            measures = evaluate_clusters(clusters, None, verbose=False)
            if measures[SSE].sum() < best[2]:
                best = (e, pts, measures[SSE])
    return best[0], best[1]


def dbscan_e_pts_selection():
    res = pd.DataFrame(columns=['e', 'pts'], index=c.ALL)
    for fn in c.ALL:
        e, pts = dbscan_hypertuning(fn)
        res.at[fn, 'e'] = e
        res.at[fn, 'pts'] = pts
    print(res)
    return res


if __name__ == "__main__":
    print(dbscan_e_pts_selection())
