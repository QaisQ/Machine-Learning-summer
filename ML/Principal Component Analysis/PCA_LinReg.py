
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 

data = pd.read_csv("Placement.csv")

data = data[[ "status", "mba_p","etest_p", "specialisation","gender", "ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s", "degree_p", "degree_t", "workex"]]
# the data has everything but salary
le = preprocessing.LabelEncoder() # to convert labels in to numberic forms
# so the machine can work with them
data.gender = le.fit_transform(list(data["gender"])) # changing the string labelings to integers, 0 and 1 in this case
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

sc = StandardScaler()
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)

pca = PCA(n_components = 1) 
  
x_train = pca.fit_transform(x_train) 
x_test = pca.transform(x_test) 
  
explained_variance = pca.explained_variance_ratio_

print (x_train)
model = svm.SVC(kernel='linear')

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)


#accuracy goes as high as 90 Percent











