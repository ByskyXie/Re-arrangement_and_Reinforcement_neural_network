import numpy as np

mat = np.zeros([12,12])
edges = [[1,2],[2,5],[2,6],[2,12],
        [3,6],[3,9],
        [4,7],[4,10],[4,11],
        [5,7],[5,8],
        [6,7],
        [8,10],
        [9,12],
        [10,11]]
edges = np.array(edges) - 1  # nodePos = nodeID-1

for e in edges:
    a, b = e
    mat[a][b]=mat[b][a]=1

np.savez("adj_mx.npz", adj_mx=mat)
