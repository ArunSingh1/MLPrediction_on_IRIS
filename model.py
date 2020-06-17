# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('iris.csv')

lb_make = LabelEncoder()
data['Species'] = lb_make.fit_transform(data['Species'])

x = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data[['Species']]

X_train, X_test, y_train, y_test= train_test_split(x,y, test_size=.3)

rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf.fit(X_train,y_train)

prediction=rf.predict(X_test) #now we pass the testing data to the trained algorithm
print('The accuracy of the RfC is:',metrics.accuracy_score(prediction,y_test))

# Saving model to disk
#pickle.dump(rf, open('model.pkl','wb'))
# Saving model to disk


# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[4, 3, 3,3]]))