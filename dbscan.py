import sys
from utils import *
import constants as c


def drop_df(df1: pd.DataFrame, df2: pd.DataFrame):
    """removes rows from df1 that are also in df2"""
    return pd.concat([df1, df2, df2]).drop_duplicates(keep=False)

def dbscan(df: pd.DataFrame, epsilon: float, min_points: int):
    # a dataframe of the distances between all datapoints
    dists = pd.DataFrame(get_euclidean_distances(df), index=df.index.values)
    # a series of the number of neighbors for each datapoint
    num_neighbors = dists[dists < epsilon].count() - 1
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
        neighborhood: pd.DataFrame = rest[dists[candidate.name] < epsilon]
        drop_df(rest, neighborhood)
        cluster = pd.DataFrame([candidate])
        while True:
            new_neighborhood = pd.DataFrame()
            ### it does what i expect, haven't had chance to debug rest. It's late, and I really just need to look at
            # it when i'm not falling asleep.
            for neighbor in neighborhood:
                pd.concat(new_neighborhood, rest[dists[neighbor] < epsilon])
            rest = rest.drop(new_neighborhood)
            cluster = pd.concat(cluster, neighborhood)
            neighborhood = new_neighborhood
            if len(new_neighborhood) == 0:
                break
        clusters.append(cluster)
    return clusters


def test():
    fn = c.FOUR_CLUSTERS
    df = parse_csv(fn)
    min_points = 2*df.shape[1]
    dbscan(df, 5, min_points)

def main():
    if len(sys.argv) != 4:
        raise TypeError(
            f'dbscan expected 3 arguments, got {len(sys.argv) - 1}')
    else:
        fn = sys.argv[1]
        epsilon = sys.argv[2]
        num_points = sys.argv[3]

    df = parse_csv(fn)
    dbscan(df, epsilon, num_points)


if __name__ == "__main__":
    test()