#!/usr/bin/env python
# coding: utf-8

# ## K-means Clustering on College Data

from sklearn.cluster import KMeans
college_df = pd.read_csv('College.csv')
college_df.head()


fig= plt.figure(figsize=(10,6))
plt.scatter(x = college_df['Outstate'],y = college_df['Grad.Rate'])
plt.title("outstate vs graduation rate")
plt.xlabel('outstate')
plt.ylabel('graduation rate')
plt.show()


college_df[college_df['Grad.Rate'] > 100]


college_df.loc[college_df['Grad.Rate'] > 100, 'Grad.Rate'] = 100


college_df[college_df['Grad.Rate'] > 100]


# Set K=2: we only want to cluster the dataset into two subgroups
kmeans = KMeans(n_clusters=2).fit(college_df[['Outstate','Grad.Rate']])


# Look at the outputs: Two cluster centers
kmeans.cluster_centers_


# Look at the outputs: Cluster labels 
kmeans.labels_


fig= plt.figure(figsize=(10,6))
# Visualise the output labels
plt.scatter(x=college_df['Outstate'],y=college_df['Grad.Rate'], c=kmeans.labels_)

# Visualise the cluster centers (black stars)
plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], 'k*',markersize=20)
plt.title('outstate vs graduation rate (k-means clustering)')
plt.xlabel('outstate')
plt.ylabel('graduation rate')
plt.show()






