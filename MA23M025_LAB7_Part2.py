#!/usr/bin/env python
# coding: utf-8

# # QUESTION-5

# WAP to plot a 3-d graph of the sine wave signal using the scatter method and normal line method. Plot them separately and specify legend.

# In[46]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate data
t = np.linspace(0, 10*np.pi, 1000)
x = np.sin(t)
y = np.cos(t)
z = t

# Scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x, y, z, c='r', marker='o', label='Sine Wave')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Scatter Plot of Sine Wave')
ax.legend()

# Line plot
ax = fig.add_subplot(122, projection='3d')
ax.plot(x, y, z, c='b', label='Sine Wave')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Line Plot of Sine Wave')
ax.legend()

plt.show()


# # QUESTION-6

# countries = {
# 
#     "1": {"Country": "New Country 1",
# 
#           "Capital": "New Capital 1",
# 
#           "Population": "123,456,789"},
# 
#     "2": {"Country": "New Country 2",
# 
#           "Capital": "New Capital 2",
# 
#           "Population": "987,654,321"},
# 
#     "3": {"Country": "New Country 3",
# 
#           "Capital": "New Capital 3",
# 
#           "Population": "111,222,333"}
# 
# }
# 
# Make a data frame using pandas from dictionary of dictionary.

# In[47]:


import pandas as pd

countries = {

"1": {"Country": "New Country 1",

      "Capital": "New Capital 1",

      "Population": "123,456,789"},

"2": {"Country": "New Country 2",

      "Capital": "New Capital 2",

      "Population": "987,654,321"},

"3": {"Country": "New Country 3",

      "Capital": "New Capital 3",

      "Population": "111,222,333"}
}

df = pd.DataFrame(countries)
print(df)


# In[48]:


df1 = pd.DataFrame(countries).T #here T gives the transfose of the data(for better view)
print(df1)


# # QUESTION-7

# StringData = ‘’’Date;Event;Cost
# 
#     10/2/2011;Music;10000
# 
#     11/2/2011;Poetry;12000
# 
#     12/2/2011;Theatre;5000
# 
#     13/2/2011;Comedy;8000
# 
#     ‘’’
# 
# Make a data frame using pandas from string.

# In[49]:


import pandas as pd

StringData = '''Date;Event;Cost
10/2/2011;Music;10000
11/2/2011;Poetry;12000
12/2/2011;Theatre;5000
13/2/2011;Comedy;8000
'''

data = [line.split(';') for line in StringData.split() if line]
df = pd.DataFrame(data[1:], columns=data[0])
print(df)


# # QUESTION-8

# Take a N X M integer array matrix with space separated elements ( N = rows and M  = columns). Your task is to print the transpose and flatten results using numpy

# In[50]:


import numpy as np

# Generate random 3x2 matrix with elements ranging from 0 to 10
matrix = np.random.randint(0, 11, size=(3, 2))
# Printing the generated matrix
print("Generated Matrix:")
print(matrix)


# In[51]:


# Printing transpose
print("\nTranspose of the matrix:")
print(matrix.T)


# In[52]:


# Printing flattened result
print("\nFlattened matrix:")
print(matrix.flatten())


# # QUESTION-9

# WAP to capitalize a column of names in a Pandas Dataframe.
# 
# Eg : Input : {'Name': ['john', 'bODAY', 'aNa', 'Peter', 'nicky'], 'Education': ['masters', 'graduate', 'graduate', 'Masters', 'Graduate'], 'Age': [27, 23, 21, 23, 24]}
# 
# Output : {'Name': ['John', 'Boday', 'Ana', 'Peter', 'Nicky'], 'Education': ['masters', 'graduate', 'graduate', 'Masters', 'Graduate'], 'Age': [27, 23, 21, 23, 24]}

# In[53]:


import pandas as pd
data = {'Name': ['john', 'bODAY', 'aNa', 'Peter', 'nicky'],
        'Education': ['masters', 'graduate', 'graduate', 'Masters', 'Graduate'],
        'Age': [27, 23, 21, 23, 24]}

df = pd.DataFrame(data)
print("Before capitalization: \n",df)
print("\n")
df['Name'] = df['Name'].str.capitalize()
print("After capitalization: \n",df)


# # QUESTION-10

#  Use the central difference method to find the first and second order derivatives of the function. Use the following function for testing the result. And also verify the result manually (Write on paper and upload jpg). Refer to section 2.5.1 of “Optimization for Engineering Design: Algorithms and Examples” by KALYANMOY DEB, 2nd edition
# 
# 
# f(x) = 3x**2 + 2x

# In[58]:


def f(x):
    return 3*x**2 + 2*x

def first_order_derivative(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def second_order_derivative(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

# Test the derivatives at x = 1 with a step size h = 0.2
x = 1
h = 0.2

first_order_result = first_order_derivative(f, x, h)
second_order_result = second_order_derivative(f, x, h)

print("First order derivative at x = 2:", first_order_result)
print("Second order derivative at x = 2:", second_order_result)


# In[ ]:




