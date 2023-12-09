import matplotlib.pyplot as plt 
import random
import numpy as np 

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data = [(1950 + i, 0.5*i + 5 + random.randint(1,10)) for i in range(70)]


model = [tup[0] for tup in data]
price = [tup[1] for tup in data]

#splitting the data into test and train 
model_train = np.array(model[:-20]).reshape(-1, 1)  # Reshape to a 2D array
model_test = np.array(model[-20:]).reshape(-1, 1)    # Reshape to a 2D array

price_train = price[:-20]
price_test = price[-20:]


#making the linear regression model 
reg = linear_model.LinearRegression()
reg.fit(model_train, price_train)

y_predict = reg.predict(model_test)
mse = mean_squared_error(price_test, y_predict)

print(f"Mean squared error: {mse}")

# weights = reg.coef_
# intercept = reg.intercept_
# print(weights, intercept)

plt.scatter(model_test, price_test)
plt.xlabel('Model')
plt.ylabel('Price in lac')
plt.scatter(model_train, price_train)
plt.plot(model_test, y_predict)
plt.show()