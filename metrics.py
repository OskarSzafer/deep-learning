import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import deep_learning as dl


df = pd.read_csv('titanic/train.csv')

X = df.drop(columns = ['Survived'])
y = df['Survived']

X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = False)

X['Age'] = X['Age'].fillna(X['Age'].median(), inplace=False)
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=False)

X = pd.get_dummies(X,columns = ['Sex','Embarked'],drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# output setup
outputDF = pd.DataFrame()
outputDF["y"] = y_test


# linear regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# rounding to 0 1
y_pred = y_pred.round()
outputDF["Linear_regression"] = y_pred

# deep learning
unbalanced = pd.concat([X_train, y_train], axis=1)
df_majority = unbalanced[unbalanced.Survived == 0]
df_minority = unbalanced[unbalanced.Survived == 1]
df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

X_upsampled = df_upsampled.drop('Survived', axis=1)
Y_upsampled = df_upsampled['Survived']

model = dl.NN(8, 14, 3, 1, 'sigmoid')
model.train(X_upsampled.to_numpy(), Y_upsampled.to_numpy(), epochs = 300)
y_pred = model.predict(X_test.to_numpy())
y_pred = y_pred.round()
outputDF["Deep_learning"] = y_pred



print(outputDF)