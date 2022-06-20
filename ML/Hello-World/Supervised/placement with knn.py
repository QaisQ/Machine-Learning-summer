#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from pandas import read_csv
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_csv("Placement.csv")

data = data[["status", "mba_p","etest_p", "specialisation","gender", "ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s", "degree_p", "degree_t", "workex"]]
# the data has everything but salary
le = preprocessing.LabelEncoder()
data.gender = le.fit_transform(list(data["gender"]))
data.ssc_b = le.fit_transform(list(data["ssc_b"]))
data.hsc_b = le.fit_transform(list(data["hsc_b"]))
data.hsc_s = le.fit_transform(list(data["hsc_s"]))
data.degree_t = le.fit_transform(list(data["degree_t"]))
data.workex = le.fit_transform(list(data["workex"]))
data.specialisation = le.fit_transform(list(data["specialisation"]))
data.status = le.fit_transform(list(data["status"]))

predict = "status"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

K = 13
model = KNeighborsClassifier(n_neighbors = K)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)



predictions = model.predict(x_test)





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




