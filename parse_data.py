import pandas as pd


def parse_csv(fn: str) -> pd.DataFrame:
    with open(fn, 'r') as f:
        h = f.readline().split(',')
    header = to_header(h)
    if header[0] == 'row_id':
        df = pd.read_csv(fn, names=header[1:], index_col=0,skiprows=1)
    else:
        df = pd.read_csv(fn, names=header, skiprows=0)
    return df


def to_header(hrow):
    res = []
    if int(hrow[0]) == 0:
        res.append('row_id')
        res += list(range(0,len(hrow)-1))
    else:
        res = list(range(0,len(hrow)))
    return tuple(res)