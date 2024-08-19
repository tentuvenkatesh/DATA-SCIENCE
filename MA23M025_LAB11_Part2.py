#!/usr/bin/env python
# coding: utf-8

# # QUESTION-2

# Using the data provided (Logistic_regression_ls.csv), plot the decision boundary (linear) using Optimization of the sigmoid function.

# In[1]:


import pandas as pd
data = pd.read_csv('Logistic_regression_ls.csv') #this data 'univariate_linear_regression.csv' has been used in our previous sir codes thats why i have used this data
print(data)


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Logistic_regression_ls.csv')

# Extract features (X1, X2) and labels (y)
X = data[['x1', 'x2']].values
y = data['label'].values

# Normalize the features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add intercept term to X
X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

# Sigmoid function
def sigmoid(z):
    # Avoid overflow by clipping values
    z_clipped = np.clip(z, -500, 500)  # Clip to prevent overflow
    return 1 / (1 + np.exp(-z_clipped))

# Optimization function with fixed alpha
def optimize_fixed_alpha(X, y, theta, alpha, num_iterations):
    m = len(y)
    for iteration in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient
    return theta

# Optimization function with gradient descent for alpha
def optimize_gradient_descent_alpha(X, y, theta, alpha, num_iterations):
    m = len(y)
    for iteration in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        alpha -= 0.01 * np.dot(gradient.T, gradient)
        theta -= alpha * gradient
    return theta

# Initialize theta and alpha
theta = np.zeros(X_with_intercept.shape[1])
alpha = 0.01
num_iters = 1000

# Choose which optimization function to use
theta_fixed_alpha = optimize_fixed_alpha(X_with_intercept, y, theta.copy(), alpha, num_iters)
theta_grad_descent_alpha = optimize_gradient_descent_alpha(X_with_intercept, y, theta.copy(), alpha, num_iters)

# Plot the decision boundaries one after another
plt.figure(figsize=(8, 5))

# Plot decision boundary using fixed alpha
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary using Fixed Alpha')

x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
X_boundary = np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()]
boundary_values = sigmoid(np.dot(X_boundary, theta_fixed_alpha)).reshape(xx1.shape)
plt.contour(xx1, xx2, boundary_values, levels=[0.5], cmap=plt.cm.Paired)
plt.show()

# Plot decision boundary using gradient descent for alpha
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary using Gradient Descent for Alpha')

boundary_values = sigmoid(np.dot(X_boundary, theta_grad_descent_alpha)).reshape(xx1.shape)
plt.contour(xx1, xx2, boundary_values, levels=[0.5], cmap=plt.cm.Paired)
plt.show()

