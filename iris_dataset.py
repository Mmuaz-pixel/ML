import pandas as pd 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


data = pd.read_csv('./iris.csv')
# print(data.shape) # gives some of data to get overview 
# print(data.head) # gives no of rows and col 
# print(data.describe()) # provides the stats of  mathematical data

# print(data.groupby('variety').size()) # size of each class (column of data we want to know about)

####### uni variable plot ###### 

# data.plot(kind='kde', subplots=True, layout=(2,2), sharex=False, sharey=False)

# kind = type of imagery (box, hist, kde, barh, line) 
# layout is how much graphs in x and how much in y 
# share x and share y is graphs share the values lines or they have seperate 

# plt.show()

#############  multi variable plot ###########

# scatter_matrix(data) # every vriable is plotted every other variable 
# plt.show()

############# Validation data set for training of model ###############

# we will  split the data into two parts. first one is used for training of model and other one is used for testing its accuracy 

array = data.values
x = array[:,0:4]
y = array[:, 4]
validation_size = 0.20
seed = 32 
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed, shuffle=True)

scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name , model in models: 
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)