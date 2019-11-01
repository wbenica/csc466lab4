import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
import math
import sys

class Datapoint:
    def __init__(self, id, data):
        self.id = id
        self.data = data        #series
    def __str__(self):
        return self.id
    def __repr__(self):
        return self.id

class Cluster:
    def __init__(self, datapoints, circumference):
        self.datapts = datapoints
        self.circ = circumference
        self.kids = []
    def __repr__(self):
        l = ""
        r = ""
        if len(self.kids) > 1:
            l = self.kids[0].__repr__()
            r = self.kids[1].__repr__()
        return ', '.join(["%s" % x.id for x in self.datapts])
        return l + " " + r + " " + ', '.join(["%s" % x.id for x in self.datapts])

#numpy wrapper class to let JSON dumps read as primitives. accounts for certain csvs
#https://stackoverflow.com/a/57915246     
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def agglomerative(dataset):
    #initialise 2d array of distances and initial rows/columns of single point clusters
    clusters = [Cluster([dataset[i]], 0) for i in range(len(dataset))]
    for i in range(len(clusters)):
        clusters[i].kids.append(dataset[i])
    distances = [ [None] * len(clusters) for i1 in range(len(clusters)) ]
    for i in range(len(clusters)):
        distances[i][i] = 0
    
    while len(clusters)>1:
        savedDist, cl1, cl2 = singleLinkDist(clusters)
        newClust = Cluster([],savedDist)
        for dp in cl1.datapts:
            newClust.datapts.append(dp)
        for dp in cl2.datapts:
          newClust.datapts.append(dp)
        newClust.kids.append(cl1)
        newClust.kids.append(cl2)
        clusters = [c for c in clusters if c != cl1 and c!= cl2]
        clusters.append(newClust)
    return newClust

def euclidianDist(dp1, dp2):
    d = 0
    for i in range(1, len(dp1.data)+1):
        d += (float(dp1.data[i]) - float(dp2.data[i]))**2
    return math.sqrt(d)
    
def singleLinkDist(clusters):
    minDist = float("inf")
    for row in range(len(clusters)-1):
        for col in range(row+1, len(clusters)):
            for pt1 in clusters[row].datapts:
                for pt2 in clusters[col].datapts:
                    dist = euclidianDist(pt1, pt2)
                    if dist < minDist:
                        minCl1 = clusters[row]
                        minCl2 = clusters[col]
                        minDist = dist
    return (minDist, minCl1, minCl2)

def completeLinkDist(clusters):
    maxDist = float("-inf")
    for row in range(len(clusters)-1):
        for col in range(row+1, len(clusters)):
            for pt1 in clusters[row].datapts:
                for pt2 in clusters[col].datapts:
                    dist = euclidianDist(pt1, pt2)
                    if dist > maxDist:
                        maxCl1 = clusters[row]
                        maxCl2 = clusters[col]
                        maxDist = dist
    return (maxDist, maxCl1, maxCl2)

def displayJSONRecursion(tree):
    node = {}
    if len(tree.kids) < 2:
        node = {"type":"leaf", "height":0, "data":tree.kids[0].id}
        return node
    node = {"type":"node"}
    node["height"] = tree.circ
    node["nodes"] = []
    l = displayJSONRecursion(tree.kids[0])
    r = displayJSONRecursion(tree.kids[1])
    node["nodes"].append(l)
    node["nodes"].append(r)
    return node

def display(tree):
    node = {"type":"root", "height":tree.circ,"nodes":[]}
    leftN = displayJSONRecursion(tree.kids[0])
    rightN = displayJSONRecursion(tree.kids[1])
    node["nodes"].append(leftN)
    node["nodes"].append(rightN)
    return node

def displayClusters(tree, threshold, clusfile):
    n = 1
    if tree.circ > threshold:
        n = displayClusters(tree.kids[0], threshold, clusfile)
        n += displayClusters(tree.kids[1], threshold, clusfile)
        return n
    print('CLUSTER:\t{',end='child: ', file=clusfile)
    print(tree.kids[0], end="", file=clusfile)
    if len(tree.kids) > 1:
        print("\n\t\t\tchild: ", end="", file=clusfile)
        print(tree.kids[1], end= "", file=clusfile)
    print("}",file=clusfile)
    return n


def main():
    fName = sys.argv[1]
    maxCirc = None
    if len(sys.argv) > 2:
        maxCirc = float(sys.argv[2])
    fIn = open(fName, "r")
    df = pd.read_csv(fName, header=None, skiprows=1)
    colsToUse = fIn.readline().strip("\n").split(",")
    colsToDrop = []
    for i in range(len(colsToUse)):
        if colsToUse[i] == '0':
            if i == 0:
                rowIds = df.iloc[:,0]
            colsToDrop.append(i)
    df.drop(df.columns[colsToDrop], axis = 1, inplace = True)

    dataset = []
    for index, datapt in df.iterrows():
        newDataPt = Datapoint(rowIds[index], datapt)
        dataset.append(newDataPt)
    tree = agglomerative(dataset)
    dict = display(tree)
    if (maxCirc):
        with open('clusters.txt', 'w') as clus:
            numClust = displayClusters(tree, maxCirc, clus)
            print("%d clusters made" % numClust, file = clus)
    with open('dendrograph.json', 'w') as outfile:
        json.dump(dict, outfile, indent =4, cls=NpEncoder)

if __name__ == '__main__':
    main()
