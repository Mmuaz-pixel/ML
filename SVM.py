import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.svm import SVC

iris = pd.read_csv('./iris.csv')
dataframe = pd.DataFrame(iris)
dataframe = dataframe.apply(lambda x: x.astype('category'))
dataframe['variety'] = dataframe['variety'].cat.codes

x = dataframe[:, 1:4]
y = dataframe['variety']

setosa_or_vertisa = (y==0) | (y==1)
x = x[setosa_or_vertisa]
y = y[setosa_or_vertisa]

model = SVC(kernel='rbf', gamma=0.5)

plt.scatter(x[:,0], x[:,1][y==0], label="class 0")
plt.legend()
plt.show()