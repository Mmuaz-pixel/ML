import numpy as np 
import pandas as pd 

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = sns.load_dataset('iris')
# print(iris.info()) # get info about the data 


# ----------- decision tree model --------------------

x = iris.drop('species', axis = 1)
y = iris['species']
le = LabelEncoder()
y = le.fit_transform(y)

decision_tree = DecisionTreeClassifier()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.5, shuffle=True)

decision_tree.fit(x_train, y_train)
predictions = decision_tree.predict(x_test)

print(accuracy_score(y_test, predictions))