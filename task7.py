import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,RidgeClassifier,Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('Task7/breast-cancer.csv')
df.head()

y=df['diagnosis']
x=df.drop('diagnosis',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=42)
log=LogisticRegression(solver="liblinear")
model=log.fit(x_train,y_train)
model.score(x_test,y_test)



fraud=df[df['diagnosis']=='M']
fraud.shape



fraud=fraud.drop('diagnosis',axis=1)
model.predict(fraud)



ridge=RidgeClassifier()
model=ridge.fit(x_train,y_train)
model.score(x_test,y_test)


vc=df['diagnosis'].value_counts()
plt.pie(vc.values,labels=vc.index,autopct="%1.1f%%")
plt.show()