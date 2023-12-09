import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('./iris.csv')
dataframe_ = pd.DataFrame(data)

# changing every column into a category 

dataframe_ = dataframe_.apply(lambda x: x.astype('category'))
dataframe = dataframe_.apply(lambda x: x.cat.codes) # gives numeric classification like 0,1,2 etc 
# print(dataframe.info())
print(dataframe_.info())
X = dataframe.drop(['variety', 'sepal.length'], axis=1)
Y = dataframe['variety']

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
NB = MultinomialNB()
NB.fit(train_x, train_y)
predict = NB.predict(test_x)

print(accuracy_score(predict, test_y))
print(confusion_matrix(predict, test_y))


# low accuracy is because of the assumption we have in naive bayes that features are independent of each other so when we encounter the case where we have good relation between features then the accuracy drops alot 