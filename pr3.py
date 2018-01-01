
# coding: utf-8

# In[28]:


import numpy as np
#import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import *
from scipy.sparse import *
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import random
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import math
from copy import deepcopy
import math
from scipy import sparse
import numpy as np
import pandas as pd
from scipy.sparse import *
from sklearn.cluster import KMeans
import datetime
from sklearn.metrics.pairwise import euclidean_distances
import random
from sklearn.metrics import pairwise_distances
from scipy import sparse
from random import randint
from sklearn.utils import shuffle
import scipy


# In[7]:


data_read = open('train.dat','r')
data = data_read.readlines()


# In[8]:


csr_rows = len(data)
csr_cols = 0 
nnz = 0 
for i in xrange(csr_rows):
    row_data = data[i].split()
    if len(row_data) % 2 != 0:
        raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(row_data)))
    nnz += len(row_data)/2
    for j in xrange(0, len(row_data), 2): 
        cid = int(row_data[j]) - 1
        if cid+1 > csr_cols:
            csr_cols = cid+1


# In[9]:


print('Number of features',csr_cols)


# In[10]:


#Get csr matrix
val = np.zeros(nnz, dtype=np.float)
ind = np.zeros(nnz, dtype=np.int)
ptr = np.zeros(csr_rows+1, dtype=np.long)
n = 0 
for i in xrange(csr_rows):
    row_data = data[i].split()
    for j in xrange(0, len(row_data), 2): 
        ind[n] = int(row_data[j]) - 1
        val[n] = float(row_data[j+1])
        n += 1
    ptr[i+1] = n 


# In[11]:


#CSR matrix creation
final_csr_nm = csr_matrix((val, ind, ptr), shape=(csr_rows, csr_cols), dtype=np.float)


# In[12]:


def csr_l2normalize(mat, copy=True, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


# In[13]:


final_csr_f = csr_l2normalize(final_csr_nm)


# # CODE K-MEANS

# In[23]:


def centroid(data):
    centroid = scipy.sparse.csr_matrix(csr_matrix.mean(data,0))
    return centroid


# In[24]:


#compute eucledian distance between points
def distance(data):
    data_center = centroid(data)
    data_length = data.shape[0]
    sim_total = 0
    print ('data_type', (data.shape))
    print ('data center', (data_center.shape))
    sim = np.sum(cosine_similarity(data_center,data))
    #sim_total = sim_total + sim
        
    #error_mean = sim_total / data_length
    error_mean = sim
    return error_mean   


# In[77]:


def kmeansAlgo(mat_sp, clus_pop, ini_mat, k):
    #
    print('1', mat_sp.shape)
    print('pop', len(clus_pop))
    #
    iter_val = 5
    no_epoch = 1
    #
    for epoch in range(no_epoch):
        print('epoch', epoch)
        #get random centroid
        num_to_select = k 
        clus_ind_len =  len(clus_pop)
        #
        random_centroid = np.random.RandomState(2)
        r_c = random_centroid.permutation(clus_ind_len)[:k]
        print('r_c',r_c)
        #
        #
        ind_val_select = []
        for i in r_c:
            ind_val_select.append(clus_pop[i])
        
        print('value', type(ind_val_select[0]) )
        #
        centroid_list_1 = ini_mat[ind_val_select[0],]
        centroid_list_2 = ini_mat[ind_val_select[1],]
        print('initial centroid1 shape', centroid_list_1.shape)
        print('initial centroid2 shape', centroid_list_2.shape)
        

        #
        
        C_old1 = np.zeros(centroid_list_1.shape)
        C_old2 = np.zeros(centroid_list_2.shape)
        print('c_old1', C_old1.shape)
        print('c_old2', C_old2.shape)
    
        #
        error = 1
        iter_cnt = 0
        #
        while (abs(error) < 2):
            iter_cnt += 1
            #cluster seperation - cls denotes cluster
            
            cls = [None] * k
            
            cls_grp = [None] * k
            cls0 = []
            #cls0_data = []
            
            cls1 = []
            #cls1_data = []
            
            #
            #pop_len = clus_pop.shape[0]
            for i in clus_pop:
                #
                temp = [None] * k
                temp[0] = init_mat[i].dot(centroid_list_1.T) 
                temp[1] = init_mat[i].dot(centroid_list_2.T) 
                    
                #split to cluster    
                if temp[0] > temp[1]:
                    cls0.append(i)
                    #cls0_data.append(ini_mat[i])
                else:
                    cls1.append(i)
                    #cls1_data.append(ini_mat[i])
                
            #update centroid for new value
            C_old1 = deepcopy(centroid_list_1)
            C_old2 = deepcopy(centroid_list_2)
            #
            cls0_data = ini_mat[cls0, :]
            cls1_data = ini_mat[cls1, :]
            for c in range(k):
                if c == 0:
                    centroid_list_1 = scipy.sparse.csr_matrix(csr_matrix.mean(cls0_data, 0))
                    #centroid_list_1 = centroid(cls0_data)
                    print('new centroid shape', centroid_list_1.shape)
                else:
                    centroid_list_2 = scipy.sparse.csr_matrix(csr_matrix.mean(cls1_data, 0))
                    print('new centroid shape', centroid_list_2.shape)
                
            #SSE COMPUTATION
            
            cls0_len = cls0_data.shape[0]
            print('cls[0]', cls0_len)
            print('centroid_list_1', centroid_list_1.shape)
            error_sum = 0
            sim = 0
            for i in range(cls0_len):
                sim_1 = 0
                sim_2 = 0
                sim_1 = (cosine_similarity(cls0_data[i], centroid_list_1))
                sim_2 = 1 - sim_1
                sim = sim + (sim_2 ** 2)
                        
                        
            sse0 = sim / cls0_len
            print('sse0', sse0)
            #####
            cls1_len = cls1_data.shape[0]
            error_sum = 0
            error = 0
            sim = 0
            for i in range(cls1_len):
                sim_1 = 0
                sim_2 = 0
                sim_1 = 
                sim_2 = 1 - sim_1
                sim = sim + (sim_2 ** 2)
                        
            sse1 = sim / cls1_len  
            print('sse1', sse1)
            #####       
            if (centroid_list_1 - C_old1).nnz == 0 and (centroid_list_2 - C_old2).nnz == 0:
                print('cluster break')
                print('cluster_data', cls0_data.shape)
                print('cluster_data2', cls1_data.shape)
                print('cls0', len(cls0))
                print('cls1', len(cls1))
                print('sse0', len(sse0))
                print('sse1', len(sse1))
                break
                
            ###
            if iter_cnt > 50:
                print('iter break')
                print('cluster_data', cls0_data.shape)
                print('cluster_data2', cls1_data.shape)
                print('cls0', len(cls0))
                print('cls1', len(cls1))
                print('sse0', len(sse0))
                print('sse1', len(sse1))
                break
                
            
    return cls0_data, cls1_data, cls0, cls1, sse0, sse1


# In[78]:


def BisectAlgo(matrix, index, k_val):
    #
    cluster = [matrix, ]
    #
    cls_grp = [index, ]
    #
    row_count = len(cluster)
    centroid_grp =  []
    cen_list_arr = []
    #
    cnt = 0
    while True:
            
            cnt += 1
            #process
            if cnt == 1:
                distance_list = [distance(i) for i in cluster]
                print('distance', distance_list)
                max_pop = np.argmax(distance_list)
                data_split = cluster.pop(max_pop)
                print('data_split_len', data_split.shape[0])
                pop_cluster = cls_grp.pop(max_pop)
                print('pop_cluste_len', len(pop_cluster))
                distance_list = []
                
            # 
            print('distance', distance_list)
            cluster_length = len(cluster)
            #
            print('cluster_length', cluster_length)
            
            #
            if cnt > 1:
                
                print('distance', distance_list)
                max_pop = np.argmax(distance_list)
                print('max-pop', max_pop)
                distan = distance_list.pop(max_pop)
                data_split = cluster.pop(max_pop)
                print('data_split_len', (data_split.shape[0]))
                pop_cluster = cls_grp.pop(max_pop)
                print('pop_cluste_len', len(pop_cluster))
                #
                print('total cluster', len(cluster))
                print('total index', len(pop_cluster))
                #print('total distance', len(distance_list))
                
            ###bisect cluster
            
            k_cluster1, k_cluster2, val_cluster1, val_cluster2, err0, err1 = kmeansAlgo(data_split, pop_cluster, matrix, 2)
            
            print('output', float(err0))
            print('cluster_length2', len(cluster))
            
            #append original cluster
            cluster.append(k_cluster1)
            cls_grp.append(val_cluster1)
            #distance_list.append(err0)
            #####
            cluster.append(k_cluster2)
            cls_grp.append(val_cluster2)
            #distance_list.append(err1)
            distance_list.append(err0)
            distance_list.append(err1)
            ####
            #print('centroid_aft_append', len(distance_list))
            ######
            
            
            row_count += 1
            #
            if row_count >= k_val:
                break
                
        
    return cluster, cls_grp
      
    


# In[79]:


#get  csr_index
len_final_csr = final_csr_f.shape[0]
csr_index = []
for i in range(len_final_csr):
    csr_index.append(i)

    
#main algorithm
final_cluster, final_cluster_index = BisectAlgo(final_csr_f, csr_index, 7)


# In[80]:


len_fin = len(final_cluster_index)

for i in range(len_fin):
    clu_len = len(final_cluster_index[i])
    #print(clu_len)


# In[81]:


len_fin = len(final_cluster_index)
output = [0] * 8580
output[1]


# In[82]:


for v in range(8580): 
    for i in range(len_fin):
        for num in final_cluster_index[i]:  
            if v == num:
                output[v] = i + 1


# In[83]:


out = np.asarray(output)


# In[84]:


out.tofile('format.dat',sep='\n', format='%d')


# log = open("result.dat", "w")
# cnt = 1
# for i in output:
#     cnt = cnt + 1
#     log.write(str(i)+'\n')  
#     
# cnt

# In[ ]:


from sklearn.metrics import silhouette_score
scorelist = []
k = []
for i in range(3,5,2):
    k.append(i)
    final_cluster, final_cluster_index = BisectAlgo(final_csr_f, csr_index, i)
    len_fin = len(final_cluster_index)
    output = [0] * 8580
    for v in range(8580): 
        for i in range(len_fin):
            for num in final_cluster_index[i]:  
                if v == num:
                    output[v] = i + 1
                
    score = silhouette_score(final_csr_f, output)
    scorelist.append(score)
    
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(k, scorelist)
plt.xlabel('k')
plt.ylabel('silhouette score')

