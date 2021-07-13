#!/usr/bin/env python
# coding: utf-8

# # <center> Assignment 2 </center>

# ## K-means Clustering on College Data

# In[46]:


from sklearn.cluster import KMeans
college_df = pd.read_csv('College.csv')
college_df.head()


# In[47]:


fig= plt.figure(figsize=(10,6))
plt.scatter(x = college_df['Outstate'],y = college_df['Grad.Rate'])
plt.title("outstate vs graduation rate")
plt.xlabel('outstate')
plt.ylabel('graduation rate')
plt.show()


# In[48]:


college_df[college_df['Grad.Rate'] > 100]


# In[49]:


college_df.loc[college_df['Grad.Rate'] > 100, 'Grad.Rate'] = 100


# In[50]:


college_df[college_df['Grad.Rate'] > 100]


# In[51]:


# Set K=2: we only want to cluster the dataset into two subgroups
kmeans = KMeans(n_clusters=2).fit(college_df[['Outstate','Grad.Rate']])


# In[52]:


# Look at the outputs: Two cluster centers
kmeans.cluster_centers_


# In[53]:


# Look at the outputs: Cluster labels 
kmeans.labels_


# In[54]:


fig= plt.figure(figsize=(10,6))
# Visualise the output labels
plt.scatter(x=college_df['Outstate'],y=college_df['Grad.Rate'], c=kmeans.labels_)

# Visualise the cluster centers (black stars)
plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], 'k*',markersize=20)
plt.title('outstate vs graduation rate (k-means clustering)')
plt.xlabel('outstate')
plt.ylabel('graduation rate')
plt.show()


# ## Findings about identified clusters
Graduation rate is the probability that they will graduate in % while outstate is the amount of people in the college that went for out-state tuition. By clustering, 2 groups have been identified. It is seen that college with around 2500 – 11300 people going for outstate tuition have randomly scattered values between 5% - 100%. Whereas for colleges that have more people going to outstate tuition, there is more of a clear distribution around 40% - 100%. Therefore, in general, there is a higher graduation rate for the college when more students attend outstate tuition. 
# In[ ]:




