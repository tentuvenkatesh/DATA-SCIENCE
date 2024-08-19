#!/usr/bin/env python
# coding: utf-8

# # QUESTION-1

# Write a program that takes coefficients A, B, C, D, and E as inputs representing a 4th degree polynomial in the form Ax^4 + Bx^3 + Cx^2 + Dx + E. Calculate the values of this polynomial for x in the range from -100 to 100, with constant discrete intervals.
# 
# Store the resulting x and y values as a NumPy array, where x represents the input values, and y represents the corresponding output values of the polynomial. Finally, use Matplotlib to plot the graph using the generated NumPy array.

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

def Calculate_poly(A,B,C,D,E, x):
    return A*x**4 +B*x**3 +C*x**2 +D*x +E

A =float(input("Enter A value :"))
B =float(input("Enter B value :"))
C =float(input("Enter C value :"))
D =float(input("Enter D value :"))
E =float(input("Enter E value :"))

x_values = np.linspace(-100,100,num=1000)
y_values = Calculate_poly(A,B,C,D,E,x_values)

plt.plot(x_values,y_values)
plt.title("Graph of the 4rth polynomial")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()


# 
# # QUESTION-2

# Suppose you have a dictionary containing information about monthly sales for different products over a period of time. The dictionary has the following structure.
# 
# sales_data = {
# 
#     'Product': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C'],
# 
#     'Month': ['Jan', 'Jan', 'Feb', 'Feb', 'Mar', 'Mar', 'Apr', 'Apr', 'Apr'],
# 
#     'Sales': [100, 150, 200, 120, 180, 220, 90, 110, 130]
# 
# }
# 
# Write a Python script to convert this dictionary into a pandas DataFrame, calculate the total sales for each product over the entire period, and then create a bar plot using matplotlib to visualize the total sales for each product.

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt

sales_data = {

'Product': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C'],

'Month': ['Jan', 'Jan', 'Feb', 'Feb', 'Mar', 'Mar', 'Apr', 'Apr', 'Apr'],

'Sales': [100, 150, 200, 120, 180, 220, 90, 110, 130]
}

df = pd.DataFrame(sales_data)
total_sales = df.groupby('Product')['Sales'].sum()
plt.bar(total_sales.index, total_sales.values)
plt.title('Total Sales for Each Product')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.show()


# # QUESTION-3

# Create visualizations for the following mathematical functions using Matplotlib:
# 
# Plot the following single-variable functions over the range 
# 
# [−10,10], and include a title and labels for the axes:
# 
# (1) y = cos(x)
# 
# (2) y = e^x
# 
# (3) y = log(x), where x>0
# 
# 
# Generate surface plots for these multi-variable functions over the range 
# 
# x=[−10,10] and y=[−10,10] , ensuring to add a title and labels for all axes:
# 
# (1) z = cos(sqrt(x^2+y^2)
# 
# (2) z = e^(-(x^2+y^2))
# 
# (3) z =  log(x^2+y^2) where x^2+y^2>0

# (1) ploting of y= cosx

# In[28]:


x_values = np.linspace(-10,10,1000)
y_values = np.cos(x_values)
plt.plot(x_values, y_values)
plt.title("Plot of y = cos(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# (2) y = e ^ x

# In[29]:


x_values = np.linspace(-10,10,100)
y_values = np.exp(x_values)
plt.plot(x_values, y_values)
plt.title("Plot of y = e^x")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# (3) y = logx

# In[30]:


x_values = np.linspace(0.1,10,1000)
y_values = np.log(x_values)
plt.plot(x_values, y_values)
plt.title("Plot of y = logx")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# Generate surface plots for these multi-variable functions over the range
# 
# x=[−10,10] and y=[−10,10] , ensuring to add a title and labels for all axes:
# 
# (1) z = cos(sqrt(x^2+y^2)
# 
# (2) z = e^(-(x^2+y^2))
# 
# (3) z = log(x^2+y^2) where x^2+y^2>0

# (1) z = cos(sqrt(x^2+y^2)

# In[31]:


from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.cos(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_title("Surface plot of z = cos(sqrt(x^2 + y^2))")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()


# (2) z = e^(-(x^2+y^2))

# In[21]:


x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_title("Surface plot of z = e^(-(x^2 + y^2))")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()


# (3) z = log(x^2+y^2) where x^2+y^2>0

# In[32]:


x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.log(X**2 + Y**2)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_title("Surface plot of z = log(x^2+y^2) where x^2+y^2>0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()


# # QUESTION-4

#  For the function J(w) = w^2 + (54/w), implement the bracketing method (choose your own a, b, n).

# In[37]:


a=0.05
b=5
n=100
x=np.linspace(a,b,n)
def eval(x):
    return  x**2+(54/x)
dx=5/100
w1=a
w2=a+dx
w3=a+2*dx
while(w3<b):
    if eval(w1)>=eval(w2) and eval(w2)<=eval(w3):
        print(f'minimum is in between {w1} and {w3}')
        flag=1
        break
    else:
        w1=w2
        w2=w3
        w3=w2+dx
if w3>=b:
    print('minimum might be at boundary point')


# In[ ]:




