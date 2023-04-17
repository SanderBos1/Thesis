import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import distance

class Pruning:

    def __init__(self, df):
       self.df = df
       self.distances = []

    # converts the whole dataframe to a cluster (the root)
    def set_root(self, df):
        centroid = random.randint(0, len(df.columns)-1)
        root = Cluster(df[df.columns[centroid]])
        df = df.drop(df.columns[centroid], axis=1)
        for column in df:
            root.set_members(df[column], distance.euclidean(root.centroid, df[column]))
        return root

    # Input: a distance threshold DThreshold and the maximum number of clusters K
    # Output: A set of clusters S, such that #S <= K
    def clustering(self, cluster, DThreshold, nrClusters):
        s = []
        for member in cluster.members:
            dist_start = float("inf")
            for i in range(len(s)):
                # print("member", member)
                nrCluster = i
                dist = distance.euclidean(s[i].centroid, member)
                # print("this is dist", dist)
                #Finds the closest existing cluster
                if dist < dist_start:
                    selected_cluster = nrCluster
                    #print(selected_cluster, "C_new")
                    dist_start = dist
                    # print("distance start", dist_start)
            #Initializes a new cluster
            if dist_start > DThreshold and len(s) < nrClusters:
                cluster = Cluster(member)
                cluster.set_members(member, 0)
                s.append(cluster)
                # print("distance allowed", DThreshold)
                # print("centroid cluster", cluster.centroid)
                # print(len(s))

            #Assigns to closest cluster
            else:
                # print("append to C_new")
                s[selected_cluster].set_members(member, dist_start)
                # print(s[selected_cluster].members)
                # print(len(s))
        return s


    def HierarchicalClustering(self, P, e, K, alpha, h_max, n_rep):
        #print(P)
        #print(P.members)
        score_begin = float("inf")
        S_start = []
        # Perform repetitions and find best clustering
        for i in range(n_rep):
            P.shuffle()
            print("shuffled", P.members)
            S = self.clustering(P, e, K)
            score = 0
            for c in S:
                #print("this is c", c)
                for i in range(1, len(c.members)):
                    dist = distance.euclidean(c.centroid, c.members[i])
                    #print(dist)
                    score = score + dist**2
                    #print(score)
                #print("score", score)
                if score < score_begin:
                    S_start = S
                    score_begin = score
        e = alpha * P.radius
        # make smaller subclusters
        for cluster in S_start:
            if len(cluster.members) > 1:
                if cluster.depth < h_max - 1:
                    self.HierarchicalClustering(cluster, e, K, alpha, h_max, n_rep)
                else:
                    self.HierarchicalClustering(cluster, 0, len(cluster.members), alpha, h_max, n_rep)
        print("this is the final object", S_start)

class Cluster:

    def __init__(self, centroid):
        self.centroid = centroid
        self.members = []
        self.depth = 0
        self.radius = 0

    def set_members(self, member, dist):
        self.members.append(member)
        if self.radius < dist:
            self.radius = dist

    def set_depth(self, depth):
        self.depth = depth

    def shuffle(self):
        np.random.shuffle(self.members)

