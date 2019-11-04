from dbscan import *


def check_pct_outliers(df, noise):
    pct_dropped = noise.shape[0] / df.shape[0]
    return pct_dropped > 0.2


def dbscan_hypertuning(fn: str):
    df, class_id = parse_csv(fn)
    # best: k, pts, SSE
    best = (0, 0, float('inf'))
    max_dist = df.max().max()
    step = min_dist = max(df.min().min() * 1.01, 1)
    num_steps = int(max_dist // min_dist + 1)
    for e in range(1, num_steps):
        for pts in range(2, df.shape[0] // 2, 2):
            clusters, noise = dbscan(df, e * step, pts)
            if len(clusters) == 0:
                continue
            if check_pct_outliers(df, noise):
                break
            measures = evaluate_clusters(clusters, None, verbose=False)
            if measures[SSE].sum() < best[2]:
                best = (e * step, pts, measures[SSE].sum())
    print(f'{fn}: e: {best[0]}, pts: {best[1]}, sse: {best[2]}')
    return best[0], best[1]


def dbscan_e_pts_selection():
    res = pd.DataFrame(columns=['e', 'pts'], index=c.ALL)
    for fn in c.DB_TESTS:
        e, pts = dbscan_hypertuning(fn)
        res.at[fn, 'e'] = e
        res.at[fn, 'pts'] = pts
    print(res)
    return res


if __name__ == "__main__":
    print(dbscan_e_pts_selection())
