#!/usr/bin/env python
# coding: utf-8

# Implement a neural network for m-samples, n-features as we discussed in class (both FP and BP) and for N layers in the hidden layer. Split the data (you can use the log. reg. data or any other one) and train your network with 70% of the data. Use 15% for validation  and test your network with the remaining 15% data. Report the evaluation metrics for varying number of layers in the network. Plot the training loss curves.

# In[1]:


import pandas as pd
df = pd.read_csv('Logistic_regression_ls.csv')
df


# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Logistic_regression_ls.csv')

# Separate features and labels
X = data[['x1', 'x2']].values
y = data['label'].values.reshape(-1, 1)

# Split data into train, validation, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def forward_propagation(X, parameters):
    # Retrieve parameters
    W1, b1, W2, b2, W3, b3 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'], parameters['W3'], parameters['b3']

    # Forward propagation
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache

def backward_propagation(X, y, cache):
    m = X.shape[0]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - y.T
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def update_parameters(parameters, gradients, learning_rate):
    parameters['W1'] -= learning_rate * gradients['dW1']
    parameters['b1'] -= learning_rate * gradients['db1']
    parameters['W2'] -= learning_rate * gradients['dW2']
    parameters['b2'] -= learning_rate * gradients['db2']
    parameters['W3'] -= learning_rate * gradients['dW3']
    parameters['b3'] -= learning_rate * gradients['db3']
    return parameters

def predict(X, parameters):
    A3, cache = forward_propagation(X, parameters)
    predictions = (A3 > 0.5)
    return predictions.astype(int)

def compute_loss(A3, y):
    m = y.shape[0]
    loss = (-1 / m) * np.sum(np.multiply(y.T, np.log(A3)) + np.multiply(1 - y.T, np.log(1 - A3)))
    return loss

def evaluate_metrics(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    mse_loss = np.mean(np.square(y_true - y_pred))
    ce_loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, mse_loss, ce_loss, precision, recall, f1_score

def model(X_train, y_train, X_val, y_val, num_iterations, learning_rate):
    input_size = X_train.shape[1]
    hidden_size1 = 5
    hidden_size2 = 5
    output_size = 1

    np.random.seed(0)
    W1 = np.random.randn(hidden_size1, input_size) * 0.01
    b1 = np.zeros((hidden_size1, 1))
    W2 = np.random.randn(hidden_size2, hidden_size1) * 0.01
    b2 = np.zeros((hidden_size2, 1))
    W3 = np.random.randn(output_size, hidden_size2) * 0.01
    b3 = np.zeros((output_size, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    costs_train = []
    costs_val = []

    for i in range(num_iterations):
        A3_train, cache_train = forward_propagation(X_train, parameters)
        cost_train = compute_loss(A3_train, y_train)
        gradients = backward_propagation(X_train, y_train, cache_train)
        parameters = update_parameters(parameters, gradients, learning_rate)

        A3_val, _ = forward_propagation(X_val, parameters)
        cost_val = compute_loss(A3_val, y_val)

        costs_train.append(cost_train)
        costs_val.append(cost_val)

        if i % 100 == 0:
            print(f"Iteration {i}: Training Loss = {cost_train}, Validation Loss = {cost_val}")

    plt.plot(costs_train, label='Training Loss')
    plt.plot(costs_val, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return parameters

# Hyperparameters
num_iterations = 1500
learning_rate = 0.01

# Train the model
parameters = model(X_train, y_train, X_val, y_val, num_iterations, learning_rate)

# Test the model
predictions_test = predict(X_test, parameters)
accuracy_test, mse_test, ce_test, precision_test, recall_test, f1_test = evaluate_metrics(y_test.flatten(), predictions_test.flatten())

print(f"Test Accuracy: {accuracy_test}")
print(f"MSE Loss: {mse_test}")
print(f"Cross Entropy Loss: {ce_test}")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1 Score: {f1_test}")

