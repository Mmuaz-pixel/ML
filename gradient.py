import numpy as np
import matplotlib.pyplot as plt 

# Generate some sample data points (you can replace this with your own dataset)
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

# Define the linear regression model: y = mx + b
# Initialize parameters (slope m and intercept b) with arbitrary values
m = 0
b = 0

# Define the learning rate and the number of iterations
learning_rate = 0.02
num_iterations = 1000

# Perform gradient descent
for i in range(num_iterations):
    # Calculate predicted values using the current parameters
    Y_pred = m * X + b

    # Calculate the gradient of the cost (mean squared error) with respect to m and b
    gradient_m = (-2 / len(X)) * np.sum(X * (Y - Y_pred))
    gradient_b = (-2 / len(X)) * np.sum(Y - Y_pred)

    # Update parameters using the gradient and learning rate
    m -= learning_rate * gradient_m
    b -= learning_rate * gradient_b

# Print the final values of m and b, which represent the best-fit line
print("Slope (m):", m)
print("Intercept (b):", b)

plt.plot(X, Y)
plt.plot(X, Y_pred)
plt.show()