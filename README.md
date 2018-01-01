# Bisecting_KMeans_Text_Clustering

Problem Statement

Implement Bisecting K-means algorithm to cluster text records

Solution

CSR matrix is created from the given text records. It is normalized and given to bisecting K-means
algorithm for dividing into cluster. In Bisecting k-means, cluster is always divided internally by 2
using traditional k-means algorithm

Methodology

- From CSR Sparse matrix CSR matrix is created and normalized
- This input CSR matrix is given to Bisecting K-means algorithm
- This bisecting k-means will push the cluster with maximum SSE to k-means for the process
of bisecting into two clusters
- This process is continued till desired cluster is obtained

Detailed Explanation

Step 1
- Input is in the form of sparse matrix, which has combination of features and its respective
values. CSR matrix is obtained by extracting features, its values and pointer variables.
- csr_matrix function is used to construct input csr matrix.
- This is then normalized using csr_normalize function

Step 2
- Data and their respective indices are taken. Indices are taken to backtrack cluster
numbers after bisecting

First Iteration:
- Entire matrix is sent into k-means for bisecting. K-means algorithm will always the input
cluster into two.
- Initially a random centroid is taken, and input data is bisected into two based on cosine
similarity between centroid and points
- Two metrics have been attempted to split the cluster one is Euclidean and other is cosine
similarity.
- Cosine similarity works better for text records compared to Euclidean metric
- Once two clusters are formed, the process is repeated by finding new cluster’s centroid.
With new centroid, data is again split into two clusters.
- Entire process is repeated till new and old centroid values remains constant. At times, there are possibility that a point can go to and fro between clusters not letting centroid to remain same. To overcome this problem, iteration count is considered. Maximum iteration is 51.
- MSE is calculated by calculating Euclidean distance between final converged centroid and its respective cluster. MSE = Euclidean distance / total cluster length

Other iterations:
- Now, from first iteration two clusters are obtained. Pop out cluster with minimum MSE
- This cluster is sent to k-means to further split into two clusters. It follows the above-mentioned steps
- This process continues till we reach maximum ‘k’ value mentioned
- In all these indexes are maintained throughout from which final clusters are obtained.
- Silhouette score is maintained for performance measure. It is calculated using the mean intra-cluster distance(a) and the mean nearest-cluster distance(b) for each sample. For a sample, it is calculated as (b-a)/max (a, b). ‘b’ is the distance between a sample and the nearest cluster that the sample is not a part of the cluster
- Attempted dimensionality reduction – ‘SVD’ by taking 2500 best components giving variance of 95 percent. This increases execution but does not give satisfactory results.
- Number of iterations are changed to increase accuracy.
- Gain between MSE were also kept as break condition. It did not give satisfactory results.

