#!/usr/bin/env python
# coding: utf-8

# # Question-1

# Implement the generalized equation for finding the gradient of m-samples, each having n-features. Also, implement the gradient descent approach assuming a constant learning rate.

# In[16]:


import numpy as np

def hypothesis(X, theta):
    return np.dot(X, theta)

def cost_gradient(X, y, theta):
    m = len(y)
    h_theta = hypothesis(X, theta)
    grad = np.dot(X.T, (h_theta - y)) / m
    return grad

def gradient_descent(theta_init, X, y, alpha=0.01, max_iterations=1000, epsilon=1e-5):
    iteration = 0
    theta = theta_init
    iterates = [list(theta)]

    while iteration < max_iterations:
        grad = cost_gradient(X, y, theta)
        if np.linalg.norm(grad) < epsilon:
            break

        theta = theta - alpha * grad
        iterates.append(list(theta))
        iteration += 1

    iterates = np.array(iterates)
    return theta, iterates


# # Question-2

# Using the code developed for problem 1, do the linear regression for the univariate problem using the attached data file univariate_linear_regression.csv. Plot the cost function (both as surface as well as contour) as well as the best fit line

# In[17]:


data = pd.read_csv('univariate_linear_regression.csv')
features = data['x'].values.reshape(-1, 1)  # Feature matrix
X = np.concatenate((np.ones((len(features), 1)), features), axis=1)
targets = data['y'].values.reshape(-1, 1)  # Target vector

# Initialize weight vector
initial_weights = np.zeros((X.shape[1], 1))

# Perform gradient descent to optimize weights
optimal_weights, weight_iterates = gradient_descent(initial_weights, X, targets)

print("Optimal weight vector is: ", optimal_weights)

# Plotting the data and the best-fit line
plt.scatter(X[:, 1], targets, color='blue', label='Data points')
plt.plot(X[:, 1], hypothesis(X, optimal_weights), color='red', label='Best fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Define cost function
def cost_function(X, y, weights):
    m = len(y)
    h_x = hypothesis(X, weights)
    cost = np.sum((h_x - y) ** 2) / (2 * m)
    return cost

# Generate values for contour plot
weight0_vals = np.linspace(-10, 10, 100)
weight1_vals = np.linspace(-10, 10, 100)
W0, W1 = np.meshgrid(weight0_vals, weight1_vals)
cost_values = np.zeros_like(W0)

# Compute cost values for contour plot
for i in range(len(weight0_vals)):
    for j in range(len(weight1_vals)):
        weights = np.array([[weight0_vals[i]], [weight1_vals[j]]])
        cost_values[i, j] = cost_function(X, targets, weights)

# Plot the cost function surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, cost_values)
ax.set_xlabel('Weight 0')
ax.set_ylabel('Weight 1')
ax.set_zlabel('Cost')
ax.set_title('Cost Function Surface Plot')
plt.show()

# Plot the contour plot of the cost function
plt.contour(W0, W1, cost_values, levels=40)
plt.plot(weight_iterates[:, 0], weight_iterates[:, 1], 'o-')
plt.xlabel('Weight 0')
plt.ylabel('Weight 1')
plt.title('Cost Function Contour Plot')
plt.show()


# # Question-3

# Using the code developed for problem 1, do the linear regression for the multivariate problem using the attached data file heart.data.csv. Plot the best fit plane for the given data. Can you also interpret the result (taking one independent variable at a time)?

# In[18]:


# Load data
data = pd.read_csv('heart.data.csv')

# Extract features and labels
feature1 = data['biking'].values.reshape(-1, 1)  # First feature
feature2 = data['smoking'].values.reshape(-1, 1)  # Second feature
X = np.concatenate((feature1, feature2), axis=1)  # Feature matrix
y = data['heart.disease'].values.reshape(-1, 1)  # Target vector

# Feature scaling using manual implementation
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term to scaled features
X_scaled = np.concatenate((np.ones((len(X_scaled), 1)), X_scaled), axis=1)

# Initialize weight vector
initial_weights = np.zeros((X_scaled.shape[1], 1))

# Perform gradient descent to optimize weights
optimal_weights, weight_iterates = gradient_descent(initial_weights, X_scaled, y)

print("Optimal weight vector:")
print(optimal_weights)

# Plotting the best-fit plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 1], X_scaled[:, 2], y, color='blue', label='Data points')
x1_range = np.linspace(min(X_scaled[:, 1]), max(X_scaled[:, 1]), 10)
x2_range = np.linspace(min(X_scaled[:, 2]), max(X_scaled[:, 2]), 10)
x1_range, x2_range = np.meshgrid(x1_range, x2_range)
y_pred = optimal_weights[0] + optimal_weights[1] * x1_range + optimal_weights[2] * x2_range
ax.plot_surface(x1_range, x2_range, y_pred, color='red', alpha=0.5)
ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')
ax.set_zlabel('Heart Disease')
plt.title('Best Fit Plane (with Feature Scaling)')
plt.legend()
plt.show()


# In[ ]:




