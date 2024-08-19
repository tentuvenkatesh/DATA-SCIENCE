#!/usr/bin/env python
# coding: utf-8

# Implement the forward propagation for a two hidden layer network for m-samples, n-features as we discussed in class. Initialize the weights randomly. Use the data from the previous labs like logistic regression. You can choose the number of neurons in the hidden layer and use sigmoid activation function.Report the evaluation metrics for the network.  Also use other non-linear activation functions like ReLU and Tanh. Report the loss using both MSE and Cross Entropy.

# In[111]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('Logistic_regression_ls.csv')

# Separate features and labels
X = data[['x1', 'x2']].values
y = data['label'].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize weights and biases
input_size = X.shape[1]
hidden_size1 = 2
hidden_size2 = 4
output_size = 1

np.random.seed(0)
W1 = np.random.randn(hidden_size1, input_size) 
b1 = np.zeros((hidden_size1, 1))
W2 = np.random.randn(hidden_size2, hidden_size1)
b2 = np.zeros((hidden_size2, 1))
W3 = np.random.randn(output_size, hidden_size2)
b3 = np.zeros((output_size, 1))

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

# Forward propagation with different activation functions
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)  # Activation function from input to first hidden layer
    
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)     # Activation function from first hidden to second hidden layer
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)  # Activation function from second hidden layer to output layer
    
    return A3
def pred_label(X,pred,T=0.3):
    labels=(pred>T).astype(int)
    return labels

# Forward propagation using different activation functions
A3_sigmoid = forward_propagation(X, W1, b1, W2, b2, W3, b3)
y_pred=pred_label(X,A3_sigmoid)

def evaluate_metrics(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    mse_loss = np.mean(np.square(y_true - y_pred))
    ce_loss = -np.mean(y_true * np.log(y_pred+(1/2)) + (1 - y_true) * np.log(1 - y_pred+ (1/2)))

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    print(TP,FP,FN,TN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, mse_loss, ce_loss, precision, recall, f1_score

# Evaluate metrics for sigmoid activation only
accuracy_sigmoid, mse_loss_sigmoid, ce_loss_sigmoid, precision_sigmoid, recall_sigmoid, f1_score_sigmoid = evaluate_metrics(y.flatten(), y_pred.flatten())

# Print evaluation metrics
print("Sigmoid Activation -\nAccuracy:", accuracy_sigmoid, "\nMSE Loss:", mse_loss_sigmoid, "\nCross Entropy Loss:", ce_loss_sigmoid)
print("Precision:", precision_sigmoid, "\nRecall:", recall_sigmoid, "\nF1 Score:", f1_score_sigmoid)


# In[112]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('Logistic_regression_ls.csv')

# Separate features and labels
X = data[['x1', 'x2']].values
y = data['label'].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize weights and biases
input_size = X.shape[1]
hidden_size1 = 2
hidden_size2 = 4
output_size = 1

np.random.seed(0)
W1 = np.random.randn(hidden_size1, input_size) 
b1 = np.zeros((hidden_size1, 1))
W2 = np.random.randn(hidden_size2, hidden_size1)
b2 = np.zeros((hidden_size2, 1))
W3 = np.random.randn(output_size, hidden_size2)
b3 = np.zeros((output_size, 1))

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

# Forward propagation with different activation functions
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X.T) + b1
    A1 = tanh(Z1)  # Activation function from input to first hidden layer
    
    Z2 = np.dot(W2, A1) + b2
    A2 = tanh(Z2)     # Activation function from first hidden to second hidden layer
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)  # Activation function from second hidden layer to output layer
    
    return A3
def pred_label(X,pred,T=0.2):
    labels=(pred>T).astype(int)
    return labels

# Forward propagation using different activation functions
A3_sigmoid = forward_propagation(X, W1, b1, W2, b2, W3, b3)
y_pred=pred_label(X,A3_sigmoid)


def evaluate_metrics(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    mse_loss = np.mean(np.square(y_true - y_pred))
    ce_loss = -np.mean(y_true * np.log(y_pred+(1/2)) + (1 - y_true) * np.log(1 - y_pred+ (1/2)))

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    print(TP,FP,FN,TN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, mse_loss, ce_loss, precision, recall, f1_score

# Evaluate metrics for sigmoid activation only
accuracy_sigmoid, mse_loss_sigmoid, ce_loss_sigmoid, precision_sigmoid, recall_sigmoid, f1_score_sigmoid = evaluate_metrics(y.flatten(), y_pred.flatten())

# Print evaluation metrics
print("Sigmoid Activation -\nAccuracy:", accuracy_sigmoid, "\nMSE Loss:", mse_loss_sigmoid, "\nCross Entropy Loss:", ce_loss_sigmoid)
print("Precision:", precision_sigmoid, "\nRecall:", recall_sigmoid, "\nF1 Score:", f1_score_sigmoid)


# In[113]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('Logistic_regression_ls.csv')

# Separate features and labels
X = data[['x1', 'x2']].values
y = data['label'].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize weights and biases
input_size = X.shape[1]
hidden_size1 = 2
hidden_size2 = 4
output_size = 1

np.random.seed(0)
W1 = np.random.randn(hidden_size1, input_size) 
b1 = np.zeros((hidden_size1, 1))
W2 = np.random.randn(hidden_size2, hidden_size1)
b2 = np.zeros((hidden_size2, 1))
W3 = np.random.randn(output_size, hidden_size2)
b3 = np.zeros((output_size, 1))

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

# Forward propagation with different activation functions
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)  # Activation function from input to first hidden layer
    
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)     # Activation function from first hidden to second hidden layer
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)  # Activation function from second hidden layer to output layer
    
    return A3
def pred_label(X,pred,T=0.7):
    labels=(pred>T).astype(int)
    return labels

# Forward propagation using different activation functions
A3_sigmoid = forward_propagation(X, W1, b1, W2, b2, W3, b3)
y_pred=pred_label(X,A3_sigmoid)

def evaluate_metrics(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    mse_loss = np.mean(np.square(y_true - y_pred))
    ce_loss = -np.mean(y_true * np.log(y_pred+(1/2)) + (1 - y_true) * np.log(1 - y_pred+ (1/2)))

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    print(TP,FP,FN,TN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, mse_loss, ce_loss, precision, recall, f1_score

# Evaluate metrics for sigmoid activation only
accuracy_sigmoid, mse_loss_sigmoid, ce_loss_sigmoid, precision_sigmoid, recall_sigmoid, f1_score_sigmoid = evaluate_metrics(y.flatten(), y_pred.flatten())

# Print evaluation metrics
print("Sigmoid Activation -\nAccuracy:", accuracy_sigmoid, "\nMSE Loss:", mse_loss_sigmoid, "\nCross Entropy Loss:", ce_loss_sigmoid)
print("Precision:", precision_sigmoid, "\nRecall:", recall_sigmoid, "\nF1 Score:", f1_score_sigmoid)

