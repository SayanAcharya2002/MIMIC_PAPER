from sklearn.cluster import KMeans
import numpy as np
from multiprocessing import Pool
import os 


def run(ncl,max_iter,sampl,clusteringID): 
    C = KMeans(n_clusters=ncl,max_iter=max_iter,n_init=1,verbose=False).fit(sampl)
    print('Clustering ID = ',clusteringID,'n_iter = ',C.n_iter_,'Inertia = ',C.inertia_)
    return C 


def kmeans_with_multiple_runs(ncl,max_iter,nclustering,sampl):
    
    num_processors = os.cpu_count() 
    p=Pool(processes = num_processors)
    # print(sampl.shape,np.average(np.sum(np.isnan(sampl),axis=1)),np.median(np.sum(np.isnan(sampl),axis=1)))
    # print(np.where(np.isnan(sampl)))
    nan_loc=np.where(np.isnan(sampl))
    sampl[nan_loc[0],nan_loc[1]]=0 # take away the specific axes with nan values
    args = [] 
    any_nan=False
    any_inf=False
    for i in range(nclustering): 
        args.append([ncl,max_iter,sampl,i])
        any_nan|=np.any(np.isnan(sampl))
        any_inf|=np.any(np.isinf(sampl))
    print(len(args),any_inf,any_nan)
    clusters = p.starmap(run,args)

    inertias = []
    for i in range(len(clusters)):
        inertias.append(clusters[i].inertia_)
    index = inertias.index(min(inertias))

    print('The best inertia = ',min(inertias))

    p.close()
    p.join()

    return clusters[index]
    

