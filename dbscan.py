import sys

import constants as c
from utils import *
from utils import drop_df


def dbscan(df: pd.DataFrame, epsilon: float, min_points: int):
    # a dataframe of the distances between all datapoints
    dists = pd.DataFrame(get_euclidean_distances(df), index=df.index, columns=df.index)
    # a series of the number of neighbors for each datapoint
    num_neighbors = dists[dists.le(epsilon)].count() - 1
    num_neighbors.index = df.index
    # a dataframe of "core" points - datapoints that have at least min_points neighbors within distance epsilon
    core = df[num_neighbors > min_points]
    # the datapoints without any neighbors
    noise = df[num_neighbors == 0]
    # removing the noise
    rest = drop_df(df, noise)
    clusters = []

    ### It's pretty much a mess from this point on.
    while core.shape[0] > 0:
        candidate = core.iloc[0]
        core = core.drop(candidate.name)
        rest = rest.drop(candidate.name)
        neighborhood: pd.DataFrame = rest[dists[candidate.name].le(epsilon)]
        rest = drop_df(rest, neighborhood)
        core = drop_df(core, neighborhood)
        cluster = pd.DataFrame([candidate])
        while True:
            new_neighborhood = pd.DataFrame()
            for neighbor in neighborhood.index.values:
                if new_neighborhood.empty:
                    new_neighborhood = rest[dists[neighbor].le(epsilon)]
                else:
                    epsilon_ = rest[dists[neighbor].le(epsilon)]
                    pd.concat([new_neighborhood, epsilon_])
            rest = drop_df(rest, neighborhood)
            core = drop_df(core, neighborhood)
            cluster = pd.concat([cluster, neighborhood])
            neighborhood = new_neighborhood
            if len(new_neighborhood) == 0:
                break
        clusters.append(cluster)
    return clusters


def test():
    fn = c.FOUR_CLUSTERS
    df = parse_csv(fn)
    min_points = 2
    clusters = dbscan(df, 15, min_points)
    for cluster in clusters:
        print(cluster)
    plot_clusters(clusters, np.array([cluster.mean() for cluster in clusters]), f'dbscan {fn}')

def main():
    if len(sys.argv) != 4:
        raise TypeError(
            f'dbscan expected 3 arguments, got {len(sys.argv) - 1}')
    else:
        fn = sys.argv[1]
        epsilon = float(sys.argv[2])
        num_points = int(sys.argv[3])

    df = parse_csv(fn)
    clusters = dbscan(df, epsilon, num_points)
    for cluster in clusters:
        print(cluster)
    plot_clusters(clusters, np.array([cluster.mean() for cluster in clusters]), f'dbscan {fn}')
    evaluate_clusters(clusters, [cluster.mean() for cluster in clusters])


if __name__ == "__main__":
    main()