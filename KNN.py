import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

dataFrame_X = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
dataFrame_Y = pd.DataFrame(cancer['target'], columns=['Cancer'])

# Standarization/normalization - we do scalling to keep values in such a way that it does not cause wrong predictions 
# The need for scalling: Imagine you have two features: Age (measured in years) and Income (measured in dollars). These two features have vastly different scales. Age might range from 0 to 100, while income could range from 20,000 to 200,000. When you use these features in machine learning algorithms, they can have unequal influences on the model, leading to biased or incorrect results. For example, income might dominate the predictions because it has larger values, even if age is also important.

scaler = StandardScaler()
scaler.fit(dataFrame_X)
# StandardScaler(with_mean=True, with_std= True, copy=True) 

#with_mean (default=True):
# with_mean is a boolean parameter that controls whether the scaler should subtract the mean from the scaled data. When set to True, it subtracts the mean from each feature, resulting in a scaled dataset with a mean of 0 for each feature. This is called mean centering.

# with_std (default=True):
# with_std is another boolean parameter that controls whether the scaler should scale the data by dividing it by the standard deviation. When set to True, it scales each feature by its standard deviation, resulting in a dataset where each feature has a standard deviation of 1.

# copy (default=True):
# The copy parameter is also a boolean option. It specifies whether the scaler should create a copy of the input data or modify the input data in place. By default, it is set to True, which means that the transform method returns a new array with the scaled data, leaving the original data unchanged.

scaled_features = scaler.transform(dataFrame_X)
df_scaled_feat = pd.DataFrame(scaled_features, columns=dataFrame_X.columns)

x_train, x_test, y_train, y_test = train_test_split(df_scaled_feat, dataFrame_Y, test_size=0.2, random_state=2, shuffle=True)

# choosing the best k value based on the error

error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    predict = knn.predict(x_test)
    error_rate.append(1 - np.mean(np.array(predict) == y_test['Cancer']))

plt.plot(range(1,40), error_rate)
plt.xlabel('K value')
plt.ylabel('error')

knn = KNeighborsClassifier(n_neighbors=np.argmin(error_rate)+1)
knn.fit(x_train, y_train)
predict = knn.predict(x_test)

radnom_forest = RandomForestClassifier(n_estimators=3, criterion='gini', random_state=1)
radnom_forest.fit(x_train, y_train)
predict_ranf = radnom_forest.predict(x_test)

print(f"Accuracy score of KNN is :{accuracy_score(predict, y_test)*100}")
print(f"Accuracy score of Random forest is :{accuracy_score(predict_ranf, y_test)*100}")

plt.show()