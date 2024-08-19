#!/usr/bin/env python
# coding: utf-8

# # QUESTION-1

# # (a ) Plot the sigmoid function. Print your interpretation on why this function is useful for a classification problem.

# In[28]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot_sigmoid():
    z = np.linspace(-10, 10, 100)
    sig = sigmoid(z)
    
    plt.plot(z, sig)
    plt.title('Sigmoid Function')
    plt.xlabel('z')
    plt.ylabel('sigma(z)')
    plt.show()

plot_sigmoid()

print("Interpretation:")
print("The sigmoid function is useful for classification problems because:")
print("1. It outputs values between 0 and 1, representing probabilities of belonging to a certain class.")
print("2. The output can be interpreted as the probability of an example belonging to the positive class.")
print("3. Its smoothness, continuity, and differentiability make it suitable for optimization algorithms like gradient descent.")
print("4. In logistic regression, the sigmoid function maps the linear combination of input features to probabilities, facilitating binary predictions.")


# # (b) Plot the log functions in the cost function individually. Print your interpretation of the log functions

# In[29]:


import numpy as np
import matplotlib.pyplot as plt

def plot_log_functions():
    z = np.linspace(0.01, 0.99, 100)  # Values of z ranging from 0.01 to 0.99 to avoid division by zero
    log_z = -np.log(z)  # Negative logarithm function
    log_one_minus_z = -np.log(1 - z)  # Negative logarithm (1 - z) function

    # Plotting the negative logarithm function
    plt.plot(z, log_z, label='-log(z)')
    plt.title('Negative Logarithm Function (-log(z))')
    plt.xlabel('z')
    plt.ylabel('-log(z)')
    plt.legend()
    plt.show()

    # Plotting the negative logarithm (1 - z) function
    plt.plot(z, log_one_minus_z, label='-log(1 - z)', color='orange')
    plt.title('Negative Logarithm (1 - z) Function')
    plt.xlabel('z')
    plt.ylabel('-log(1 - z)')
    plt.legend()
    plt.show()
plot_log_functions()

print("Interpretation:")
print("\nThe two log functions plotted above are commonly used in logistic regression cost functions:")
print("\nThe negative logarithm function (-log(z)) penalizes predictions that are close to 0.")
print("\nIt increases rapidly as z approaches 0, indicating a high penalty for predicting a positive class when the true label is negative.")
print("\nThe negative logarithm (1 - z) function (-log(1 - z)) penalizes predictions that are close to 1.")
print("\nIt increases rapidly as z approaches 1, indicating a high penalty for predicting a negative class when the true label is positive.")


# # (c) Using your own data for a single feature problem, and assuming linear regression problem, plot the cost function and the corresponding contours. Also, using cross entropy as the cost function, plot it as well as its contours.

# In[11]:


import pandas as pd
data = pd.read_csv('univariate_linear_regression.csv') #this data 'univariate_linear_regression.csv' has been used in our previous sir codes thats why i have used this data
print(data)


# In[12]:


import numpy as np

# Load the data
data = np.genfromtxt('univariate_linear_regression.csv', delimiter=',', skip_header=1)
X = data[:, 0]
y = data[:, 1]

# Define threshold
threshold = np.mean(y)

# Convert y data into 0 or 1
y_binary = np.where(y > threshold, 1, 0)

# Stack X and y_binary horizontally
updated_data = np.column_stack((X, y_binary))

# Print the updated dataset
print("x, y_binary")
for row in updated_data:
    print(", ".join(map(str, row)))


# In[13]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.genfromtxt('univariate_linear_regression.csv', delimiter=',', skip_header=1)
X = data[:, 0]
y = data[:, 1]

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.theta0_range = np.linspace(-20, 20, 100)  # Adjust range for better visualization
        self.theta1_range = np.linspace(-20, 20, 100)  # Adjust range for better visualization
        
    def cost_function(self, theta0, theta1):
        m = len(self.y)
        predictions = theta0 + theta1 * self.X
        cost = (1/(2*m)) * np.sum((predictions - self.y)**2)
        return cost
    
    def plot_cost_function(self):
        costs = np.zeros((len(self.theta0_range), len(self.theta1_range)))
        for i, theta0 in enumerate(self.theta0_range):
            for j, theta1 in enumerate(self.theta1_range):
                costs[i, j] = self.cost_function(theta0, theta1)
                
        plt.figure(figsize=(10, 5))
        plt.contour(self.theta0_range, self.theta1_range, costs, levels=20)
        plt.xlabel('theta0')
        plt.ylabel('theta1')
        plt.title('Cost Function for Linear Regression')
        plt.colorbar(label='Cost')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.theta0_range, [self.cost_function(theta0, 0) for theta0 in self.theta0_range], label='Cost function for theta1=0')
        plt.xlabel('theta0')
        plt.ylabel('Cost')
        plt.title('Cost Function for Linear Regression (theta1=0)')
        plt.legend()
        plt.show()

class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.theta0_range = np.linspace(-20, 20, 100)  # Adjust range for better visualization
        self.theta1_range = np.linspace(-20, 20, 100)  # Adjust range for better visualization
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, theta0, theta1):
        m = len(self.y)
        z = theta0 + theta1 * self.X
        predictions = self.sigmoid(z)
        epsilon = 1e-10  # Small constant to avoid division by zero
        cost = -(1/m) * np.sum(self.y * np.log(predictions + epsilon) + (1 - self.y) * np.log(1 - predictions + epsilon))
        return cost
    
    def plot_cost_function(self):
        costs = np.zeros((len(self.theta0_range), len(self.theta1_range)))
        for i, theta0 in enumerate(self.theta0_range):
            for j, theta1 in enumerate(self.theta1_range):
                costs[i, j] = self.cost_function(theta0, theta1)
                
        plt.figure(figsize=(10, 5))
        plt.contour(self.theta0_range, self.theta1_range, costs, levels=20)
        plt.xlabel('theta0')
        plt.ylabel('theta1')
        plt.title('Cost Function for Logistic Regression (Cross-Entropy Loss)')
        plt.colorbar(label='Cost')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.theta0_range, [self.cost_function(theta0, 0) for theta0 in self.theta0_range], label='Cost function for theta1=0')
        plt.xlabel('theta0')
        plt.ylabel('Cost')
        plt.title('Cost Function for Logistic Regression (theta1=0)')
        plt.legend()
        plt.show()

# Linear Regression
linear_reg = LinearRegression(X, y)
linear_reg.plot_cost_function()

# Logistic Regression
logistic_reg = LogisticRegression(X, y)
logistic_reg.plot_cost_function()

