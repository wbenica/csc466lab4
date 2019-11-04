import sys

import constants as c
from utils import *
from utils import drop_df


def dbscan(df: pd.DataFrame, epsilon: float, min_points: int) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    # a dataframe of the distances between all datapoints
    dists = pd.DataFrame(get_euclidean_distances(df), index=df.index, columns=df.index)
    # a series of the number of neighbors for each datapoint
    num_neighbors: pd.Series = dists[dists.le(epsilon)].count() - 1
    num_neighbors.index = df.index
    idx = num_neighbors.sort_values(ascending=False).index
    # a dataframe of "core" points - datapoints that have at least min_points neighbors within distance epsilon
    core = df.loc[idx][num_neighbors > min_points]
    # the datapoints without any neighbors
    noise = df[num_neighbors == 0]
    # removing the noise
    rest = drop_df(df, noise)
    clusters = []

    while core.shape[0] > 0:
        curr_candid = core.iloc[0]
        core = core.drop(curr_candid.name)
        if curr_candid.name in rest:
            rest = rest.drop(curr_candid.name)
        curr_neighborhood: pd.DataFrame = pd.DataFrame([curr_candid])
        cluster = pd.DataFrame()
        while True:
            # neighbors of neighbors
            new_neighborhood = pd.DataFrame()
            # build new neighborhood from current neighborhood
            for neighbor in curr_neighborhood.index.values:
                new_neighbor_mask = dists[neighbor].le(epsilon)
                rest_mask = pd.Series(df.index.isin(rest.index), index=df.index)
                if new_neighborhood.empty:
                    new_neighborhood = df[rest_mask]
                    new_neighborhood = new_neighborhood[new_neighbor_mask]
                else:
                    new_neighbors = df[rest_mask]
                    new_neighbors = new_neighbors[new_neighbor_mask]
                    new_neighborhood = pd.concat([new_neighborhood, new_neighbors]).drop_duplicates(keep=False)
            core = drop_df(core, new_neighborhood)
            rest = drop_df(rest, new_neighborhood)
            new_neighborhood = drop_df(new_neighborhood, curr_neighborhood)
            new_neighborhood = drop_df(new_neighborhood, noise)
            if cluster.empty:
                cluster = curr_neighborhood
            else:
                cluster = pd.concat([cluster, curr_neighborhood]).drop_duplicates(keep=False)
            curr_neighborhood = new_neighborhood
            if len(new_neighborhood) == 0:
                break
        clusters.append(cluster)
    return clusters, noise.append(rest)


def main():
    if len(sys.argv) != 4:
        raise TypeError(
            f'dbscan expected 3 arguments, got {len(sys.argv) - 1}')
    else:
        fn = sys.argv[1]
        epsilon = float(sys.argv[2])
        num_points = int(sys.argv[3])

    df, class_id = parse_csv(fn)
    clusters, noise = dbscan(df, epsilon, num_points)
    sfn = strip_file_path(fn)
    evaluate_clusters(clusters, [cluster.mean() for cluster in clusters], verbose=True, outliers=noise)
    evaluate_classes(clusters, class_id)
    plot_clusters(clusters, np.array([cluster.mean() for cluster in clusters]), f'dbscan {sfn}')


if __name__ == "__main__":
    main()
