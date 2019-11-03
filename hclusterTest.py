import sys
import pandas as pd
import scipy
import plotly.figure_factory as ff
import numpy as np
np.random.seed(1)

def main():
    fName = sys.argv[1]
    rowIds = None
    fIn = open(fName, "r")
    df = pd.read_csv(fName, header=None, skiprows=1)
    colsToUse = fIn.readline().strip("\n").split(",")
    colsToDrop = []
    for i in range(len(colsToUse)):
        if colsToUse[i] == '0':
            if i == 0:
                rowIds = df.iloc[:,0]
            colsToDrop.append(i)
    df.drop(df.columns[colsToDrop], axis = 1, inplace=True)
    fig = ff.create_dendrogram(df)
    fig.update_layout(width=800, height=500)
    fig.show()

if __name__ == '__main__':
    main()
