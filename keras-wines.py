#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('wines.csv')


# In[3]:


df.info()


# In[4]:


y = df['Class']


# In[5]:


y


# In[6]:


y.value_counts()


# In[7]:


y_cat = pd.get_dummies(y)


# In[8]:


y


# In[9]:


df.columns


# In[10]:


X = df.drop('Class' , axis =1)


# In[11]:


X


# In[12]:


import seaborn as sns


# In[14]:


sns.scatterplot(x = 'Alcohol' , y=y ,data =df)


# In[15]:


from keras.models import Sequential 


# In[16]:


model =Sequential()


# In[17]:


X.info()


# In[18]:


X.shape


# In[19]:


y_cat.shape


# In[20]:


from keras.layers import Dense


# In[22]:


model.add(Dense(units =5 ,input_shape = (13,),
                activation = 'relu',
                kernel_initializer='he_normal'))


# In[23]:


model.summary()


# In[24]:


model.add(Dense(units =8,
                activation = 'relu',
                kernel_initializer='he_normal'))


# In[25]:


model.summary()


# In[26]:


model.add(Dense(units =2,
                activation = 'relu',
                kernel_initializer='he_normal'))


# In[27]:


model.add(Dense (units=3, activation ='softmax'))


# In[28]:


model.summary()


# In[30]:


from keras.optimizers import RMSprop


# In[31]:


model.compile( optimizer=RMSprop(learning_rate =0.01),
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])


# In[32]:


model.layers[0].input


# In[33]:


model.layers[3].input


# In[34]:


model.layers[2].input


# In[35]:


model.layers[2].output


# In[36]:


model.fit(X,y_cat, epochs=100)


# In[37]:


import keras.backend as K


# In[38]:


K.clear_session()


# In[39]:


model.get_weights()


# In[40]:


model.save('modelsavewine.h5')


# In[ ]:




