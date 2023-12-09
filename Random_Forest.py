import pandas as pd 
import numpy as np 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dataSet = sns.load_dataset('penguins')
dataSet = dataSet.dropna()
# print(dataSet.info())

#-----------------------Feature engineering on dataset ------------------------------

# we notice that there are object data types in our data so we will do feature engineering on it 

sex = pd.get_dummies(dataSet['sex'], drop_first=True)
island = pd.get_dummies(dataSet['island'], drop_first=True)

dataSet = pd.concat([dataSet, sex, island], axis=1)
dataSet = dataSet.drop('sex', axis=1)
dataSet = dataSet.drop('island', axis=1)


X = dataSet.drop('species', axis=1)
Y = dataSet['species']

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1, shuffle=True, test_size=0.2)
classifier = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=1) #n_estimator is the number of trees used
classifier.fit(x_train, y_train)

predict = classifier.predict(x_test)

print(confusion_matrix(y_test, predict))
print(f"Accuracy score: {accuracy_score(y_test, predict)*100}")