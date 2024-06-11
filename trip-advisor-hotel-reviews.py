#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("C:\\Users\\Masooma\\Downloads\\archive\\tripadvisor_hotel_reviews.csv")


# In[3]:


df.head()


# In[4]:


df['Rating'].info()


# In[5]:


df.info()


# In[6]:


df['Rating'].mode()


# In[7]:


df['Rating'].unique()


# In[8]:


df['Rating'].value_counts()


# In[9]:


#As the data is imbalanced, we do undrsampling for 5 star reviews


# In[10]:


df_neg=df.loc[df['Rating']<3]
df_neg=df_neg.reset_index(drop=True)
len(df_neg)


# In[11]:


df_five=df.loc[df['Rating']==5]
df_five=df_five.reset_index(drop=True)


# In[12]:


df_pos=df_five.loc[:len(df_neg)]
len(df_pos)


# In[13]:


df_all=pd.concat([df_neg,df_pos],axis=0)
df_all=df_all.reset_index(drop=True)
df_all


# ### Sentiment Column

# In[14]:


import numpy as np
df_all['Sentiment']=np.where(df_all['Rating']==5,'Positive','Negative')
df_all=df_all.reset_index(drop=True)


# In[15]:


df_all=df_all.sample(frac=1)
df_all=df_all.reset_index(drop=True)
df_all


# ### Text Cleaning

# In[16]:


import nltk  
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()


# In[17]:


corpus=[]
for i in range(len(df_all)):
    rp=re.sub('[^a-zA-Z]'," ",df_all['Review'][i])
    rp=rp.lower()
    rp=rp.split()
    rp=[ps.stem(word) for word in rp if not word in set(stopwords.words('english'))]
    rp=' '.join(rp)
    corpus.append(rp)


# In[18]:


corpus


# In[19]:


X=corpus
y=df_all["Sentiment"]
y


# In[20]:


len(X)


# ### Train|Test Split

# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
y_train


# ### Vectorization

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_trainvec=cv.fit_transform(X_train)
X_testvec=cv.transform(X_test)




# ### Modeling

# ### Naive Bayes

# In[23]:


from sklearn.naive_bayes import MultinomialNB
NB=MultinomialNB()

NB.fit(X_trainvec,y_train)


# In[24]:


#print("X_trainvec shape:", X_trainvec.shape)
#print("y_train shape:", y_train.shape)


# In[25]:


ypred_train=NB.predict(X_trainvec)
ypred_test=NB.predict(X_testvec)


# In[26]:


from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import cross_val_score
print('Train Accuracy: ',accuracy_score(y_train,ypred_train))
print('Test Accuracy: ',accuracy_score(y_test,ypred_test))
print('CV: ',cross_val_score(NB,X_trainvec,y_train,cv=5).mean())


# In[27]:


f1_score(y_test,ypred_test,average=None)


# ### SVM

# In[28]:


from sklearn.svm import SVC
sv=SVC()

sv.fit(X_trainvec,y_train)


# In[29]:


ypred_train=sv.predict(X_trainvec)
ypred_test=sv.predict(X_testvec)


# In[30]:


print('svTrain Accuracy: ',accuracy_score(y_train,ypred_train))
print('svTest Accuracy: ',accuracy_score(y_test,ypred_test))
print('svCV: ',cross_val_score(NB,X_trainvec,y_train,cv=5).mean())


# In[31]:


f1_score(y_test,ypred_test,average=None)


# ### Test on new data

# ##### Instance 1

# In[32]:


rev1=['Beautiful place,wonderful view.Customer service is also up to the mark']
rev1_vec=cv.transform(rev1)
sv.predict(rev1_vec)


# In[33]:


NB.predict(rev1_vec)


# ##### Instance 2

# In[34]:


rev2=['Dirty pillows,stinky rooms']
rev2_vec=cv.transform(rev2)
sv.predict(rev2_vec)


# In[35]:


NB.predict(rev2_vec)


# In[ ]:




