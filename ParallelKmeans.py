"""
<num of samples> <num of clusters> <num of mappers>
For plotting, num of cluster has to be between 0 and 7
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import timeit as t
import sys
from multiprocessing import Process, Pool, Manager

num_samples = int(sys.argv[1])

# Pre-processing the data sample
X = pd.read_csv('samples/'+str(num_samples)+'_samples') # X are samples
del X['Unnamed: 0']
X = pd.DataFrame.as_matrix(X)

# Step 1 - Ask the desired 'k'
k = int(sys.argv[2]) # For plotting, k = [0 .. 7]
dim = 2
num_mappers = int(sys.argv[3])

# Step 2 - Randonly assigns k points to be the cluster's center 
cluster_points = np.random.random((k, dim)) # give clusters initial points
centroid = [c for c in cluster_points]
cost_prev = 0.
convergence = False

# Globals
final_clusters = dict((key, []) for key in xrange(k))
clusters = dict((key, []) for key in xrange(k))#

def euclidean_distance(u, v):
    return math.sqrt(sum([(u[i]-v[i])**2 for i in xrange(dim)]))

def distance(u, v):
    return sum([(u[i]-v[i])**2 for i in xrange(dim)])

def wcv(clusters, centroid):
    # Calculate the sum of squared errors
    # Within Cluster Variation
    sse = 0.
    for i, samples in clusters.iteritems():
        sse += sum([distance(p, centroid[i]) for p in samples])
    return sse

def bcv(points):
    # Between Cluster Variation
    dist = 0.
    
    # This implementation reduces the matrix complexity by 75%
    even = [i for i in xrange(points.shape[0]) if i%2]
    odd = [i for i in xrange(points.shape[0]) if not i%2]

    for line in even:
        for column in odd:
            dist += sum([(points[line][i]-points[column][i])**2 for i in xrange(dim)])
    return dist
 
def mapper(sample):
    if sample != []:
        # Step 3 - For each sample, find the nearest cluster
        dist = [] # distances from sample to each cluster
        for cluster_pos in cluster_points:#
            dist.append(euclidean_distance(sample, cluster_pos))#
        owner = dist.index(min(dist))#
        return owner

def kmeans(X, k, c):
    global final_clusters, cluster_points, centroid, cost_prev

    while True:
        sse = 0.
        cost = 0.

        # Clusters to be filled
        clusters = dict((key, []) for key in xrange(k))#
        
        #mapper        
        pool = Pool(processes=num_mappers)
        i = 0
        for mapper_id in xrange(num_mappers):
            block_size = float(num_samples) / float(num_mappers)
            block_init = int(mapper_id) * int(block_size)
            block_end = int(block_init) + int(block_size)
            
            if mapper_id == (3):
                block_end = int(num_samples)
                
            ret = pool.map(mapper, X[block_init:block_end])
            
            for _, cluster_key in enumerate(ret):
                try:
                    clusters[cluster_key].append(X[i])
                except IndexError:
                    pass
                i += 1
        del pool
        #
        
        sse = wcv(clusters, cluster_points)#
        bcvar = bcv(cluster_points)
        cost = bcvar / sse
        
        # Step 4 - Find the centroid for each group of samples inside a cluster
        for key, samples in clusters.iteritems():
            if samples != []:
                centroid.append(np.sum(samples, axis=0) / np.array(samples).shape[0])
                cluster_points = np.array(centroid[len(centroid)-k:])
        
        if cost == cost_prev:   
            final_clusters.update(clusters)
            break
        else:
            cost_prev = cost

# Main
begin = t.default_timer()
kmeans(X, k, c)
end = t.default_timer()
print 'parallel time elapsed ', end - begin

colors={1:'blue', 2:'red', 3:'green', 4:'yellow', 5:'orange',
        6:'pink', 7:'purple', 8:'', 9:'', 10:''}

for key, group in final_clusters.iteritems():
    M = np.asarray(group)
    plt.scatter(M[:, 0], M[:, 1], c=colors[key+1])
    plt.scatter(cluster_points[key][0], cluster_points[key][1], c=colors[key+1], marker='x')
plt.savefig('results/par_'+str(num_samples)+'_'+str(k))
