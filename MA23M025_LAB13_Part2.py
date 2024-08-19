#!/usr/bin/env python
# coding: utf-8

# # QUESTION:

# Taking any two classes from the above data, add labels to them (0 or 1) and create a new csv file. Split the data into Train / Test set as 70/30. (a) Plot the decision boundary using the developed logistic regression code (either with or without regularization) from one of your previous labs. (b) Evaluate the metrics such as Precision, Recall, F1-Score and Accuracy on the test data without using any library.

# In[37]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('all_points.csv')

# Assign labels 0 and 1 to two classes
class_0 = data.iloc[:10]  # First set of points
class_1 = data.iloc[20:]  # Second set of points

class_0['label'] = 0
class_1['label'] = 1

# Concatenate the labeled data
labeled_data = pd.concat([class_0, class_1])

# Split the data into Train/Test sets (70/30 split)
train_data, test_data = train_test_split(labeled_data, test_size=0.3, random_state=42)

# Save the labeled and split data to a new CSV file
train_data.to_csv('train_data1.csv', index=False)
test_data.to_csv('test_data1.csv', index=False)


# In[38]:


df = pd.read_csv('all_points.csv')
print("Orginal Data we got from Last lab is: \n",df)
df1 = pd.read_csv('train_data1.csv')
print("Train data after the spliting(70|30) of Orginal data: \n",df1)
df2 = pd.read_csv('test_data1.csv')
print("Test data after the spliting(70|30) of Orginal data: \n",df2)


# In[41]:


import numpy as np
import matplotlib.pyplot as plt

# Load the training data
train_data = pd.read_csv('train_data1.csv')

# Extract features (x) and labels (y)
X_train = train_data[['x', 'y']].values
y_train = train_data['label'].values

# Add bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Initialize weights
np.random.seed(0)
theta = np.random.rand(X_train.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent
alpha = 0.01  # learning rate
epochs = 1000
for i in range(epochs):
    z = np.dot(X_train, theta)
    h = sigmoid(z)
    gradient = np.dot(X_train.T, (h - y_train)) / y_train.size
    theta -= alpha * gradient

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(train_data[train_data['label'] == 0]['x'], train_data[train_data['label'] == 0]['y'], color='blue', label='Class 0')
plt.scatter(train_data[train_data['label'] == 1]['x'], train_data[train_data['label'] == 1]['y'], color='red', label='Class 1')
x_vals = np.array([min(X_train[:, 1]), max(X_train[:, 1])])
y_vals = -(theta[0] + theta[1] * x_vals) / theta[2]  # Decision boundary equation: theta0 + theta1*x + theta2*y = 0
plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.show()


# In[40]:


# Load the test data
test_data = pd.read_csv('test_data1.csv')

# Extract features (x) and labels (y) for test set
X_test = test_data[['x', 'y']].values
y_test = test_data['label'].values

# Add bias term
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Predictions
predictions = sigmoid(np.dot(X_test, theta))
predicted_labels = np.round(predictions)

# Evaluation metrics
TP = np.sum((predicted_labels == 1) & (y_test == 1))
FP = np.sum((predicted_labels == 1) & (y_test == 0))
TN = np.sum((predicted_labels == 0) & (y_test == 0))
FN = np.sum((predicted_labels == 0) & (y_test == 1))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
print(f"Accuracy: {accuracy:.2f}")

