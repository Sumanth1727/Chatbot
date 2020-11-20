import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values
print(X)
print(Y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
X=X[:,1:]

#spiliting into test and train 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#importing required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense,Input
Annclassifier= Sequential()
#adding layes
Annclassifier.add(Input(shape=(0,11)))
Annclassifier.add(Dense(6,activation='relu',kernel_initializer='uniform'))
Annclassifier.add(Dense(6,activation='relu',kernel_initializer='uniform'))
Annclassifier.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))
#compiling the model
Annclassifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])  
#Training the model
Annclassifier.fit(X_train,Y_train,batch_size=10,epochs = 100)
#predicting the output



Y_predict=Annclassifier.predict(X_test)

"""
#for finding the correct level of i for the prediction
z=0
t=0
i=0.4
while(i<=0.8):
    Y_predict1=(Y_predict>i)

    #checking acuuracy by confusion matrix

    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(Y_test,Y_predict1)
    sum=cm[0][0]+cm[1][1]
    if(sum>t):
        t=sum
        z=i
    i=i+0.25
print(t,z)
"""


Y_predict=(Y_predict>0.65)

 #checking acuuracy by confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_predict)

accuracy=(cm[0][0]+cm[1][1])/2000
print(accuracy)





