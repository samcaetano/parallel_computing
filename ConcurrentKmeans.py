"""
args: <number of samples> <num of clusters> <number of mappers>
For plotting, number of cluster has to be between 0 and 7
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import timeit as t
import sys
from threading import Thread, Lock, Condition

# User's args
num_samples = int(sys.argv[1])
num_mappers = int(sys.argv[3])

# Pre-processing the data sample
X = pd.read_csv('samples/'+str(num_samples)+'_samples') # X are samples
del X['Unnamed: 0']
X = pd.DataFrame.as_matrix(X)

# Step 1 - Ask the desired 'k'
k = int(sys.argv[2])
dim = 2

# Step 2 - Randonly assigns k points to be the cluster's center 
cluster_points = np.random.random((k, dim)) # give clusters initial points
centroid = [c for c in cluster_points]
cost_prev = 0.
convergence = False

# Globals
colors={1:'blue', 2:'red', 3:'green', 4:'yellow', 5:'orange',
        6:'pink', 7:'purple', 8:'', 9:'', 10:''} 
clusters = dict((key, []) for key in xrange(k))
flag_counter = 0
final_clusters = dict((key, []) for key in xrange(k))
lock = Lock()
conditions = [Condition() for _ in range(num_mappers)]

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
    
    even = [i for i in xrange(points.shape[0]) if i%2]
    odd = [i for i in xrange(points.shape[0]) if not i%2]

    for line in even:
        for column in odd:
            dist += sum([(points[even][0][i]-points[odd][0][i])**2 for i in xrange(dim)])
    return dist
    
def map(thread_id, lock, conditions, X):
    global clusters, cluster_points
    global flag_counter, convergence
    
    block_size = float(num_samples) / float(num_mappers)
    block_init = int(thread_id) * int(block_size)
    block_end = int(block_init) + int(block_size)
    
    if thread_id == (num_mappers-1):
        block_end = int(num_samples)
    
    while not convergence:
        
        conditions[thread_id].acquire()
        
        #print 'scanner-%d working...' % thread_id, 
        
        
        # Clusters to be filled
        local_clusters = dict((key, []) for key in xrange(k)) # private
        
        # Step 3 - For each sample, find the nearest neighbor
        for sample in X[block_init:block_end]:
            dist = [] 
            for cluster_pos in cluster_points:
                dist.append(euclidean_distance(sample, cluster_pos)) # private
            owner = dist.index(min(dist)) # private
            local_clusters[owner].append(sample) # private
        
        #print 'scanner-%d writing on clusters' % thread_id
        lock.acquire()
        for key, lst in local_clusters.iteritems():
            clusters[key] += local_clusters[key]
        
        flag_counter += 1
        lock.release()
        #print 'scanner-%d waiting...' % thread_id
        conditions[thread_id].wait()
        conditions[thread_id].release()

def reduce(thread_id, lock, conditions, X, k):
    global final_clusters, clusters, cluster_points, centroid
    global convergence, cost_prev, flag_counter

    while not convergence:    
        if flag_counter < num_mappers:
            pass
        
        else:
            for scanner_id in xrange(num_mappers):
                conditions[scanner_id].acquire()
            
            sse = wcv(clusters, cluster_points)    
            bcvar = bcv(cluster_points) # private
            cost = bcvar / sse # private
            
            # Step 4 - Find the centroid for each group of samples inside a cluster
            for key, samples in clusters.iteritems():
                if samples != []:
                    centroid.append(np.sum(samples, axis=0)/np.array(samples).shape[0])
                    cluster_points = np.array(centroid[len(centroid)-k:])
            if cost == cost_prev:
                for key, value in clusters.iteritems():   
                    for elem in value:
                        final_clusters[key].append(elem)
                convergence = True
                for scanner_id in xrange(num_mappers):
                    conditions[scanner_id].notify()
                    conditions[scanner_id].release()
                break
            else:
                cost_prev = cost

                clusters = dict((key, []) for key in xrange(k)) # consumer resets
                flag_counter = 0 # consumer resets
                
                for scanner_id in xrange(num_mappers):
                    conditions[scanner_id].notify()
                    conditions[scanner_id].release()    

# Main

begin = t.default_timer()
mappers=[]
for i in xrange(num_mappers):
    mapper = Thread(target=map, args=(i, lock, conditions, X,))
    mappers.append(mapper)
reducer = Thread(target=reduce, args=(i, lock, conditions, X, k,))
# Fork
for mapper in mappers:
    mapper.start()
reducer.start()
# Join
for mapper in mappers:
    mapper.join()
reducer.join()
end = t.default_timer()
print 'concurrent time elapsed ', end - begin

for key, group in final_clusters.iteritems():
    M = np.asarray(group)
    plt.scatter(M[:, 0], M[:, 1], c=colors[key+1])
    plt.scatter(cluster_points[key][0], cluster_points[key][1], c=colors[key+1], marker='x')
plt.savefig('results/par'+str(num_mappers)+'_'+str(num_samples)+'_'+str(k)+'-concurrent')
