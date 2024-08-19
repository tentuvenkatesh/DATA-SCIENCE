#!/usr/bin/env python
# coding: utf-8

# # QUESTION-1

# For the function J(w) = w^2 + (54/w), implement the following methods: (a) Use the bracketed value (that you got in the last lab) to get to the critical point employing interval halving method and (b) identify the critical point using Newton-Raphson method and (c) verify the result manually using the optimality criteria (post this write-up as well in .jpg/.png etc).

# In[3]:


#Previous lab code for implimenting this function
import numpy as np
a=0.05
b=5
n=100
x=np.linspace(a,b,n)
def eval(x):
    return  x**2+(54/x)
def bracket_search(eval,a,b,n):
    dx=(b-a)/n
    w1=a
    w2=a+dx
    w3=a+2*dx
    while(w3<b):
        if eval(w1)>=eval(w2) and eval(w2)<=eval(w3):
            print(f'minimum is in between {w1} and {w3}')
            return [w1,w3]
            flag=1
            break
        else:
            w1=w2
            w2=w3
            w3=w2+dx
        if w3>=b:
            print('minimum might be at boundary point')
    
interval = bracket_search(eval,a,b,n)
interval


# (a) Use the bracketed value (that you got in the last lab) to get to the critical point employing interval halving method 

# In[4]:


def J(w):
    return w**2 + (54/w)

def Interval(J,a, b, n, tolerance=1e-8):
    interval = bracket_search(J,a,b,n)
    a = interval[0] 
    b = interval[1]
    t = b-a
    while abs(t) > tolerance:
        x1 = a + t/4
        x3 = b - t/4
        x2 = (a + b) / 2
        fa, fb, fc = J(x1), J(x2), J(x3)
        
        if fa < fb:
            b = x2
            x2 = x1
        elif fc < fb:
            a = x2
            x2 = x3
        else:
            a = x1
            b = x3
        t = b - a
        
    # Return the midpoint of the final interval
    return a

a = 0.5  #2.9499999999999975
b =5   # 3.049999999999997
n = 100

# Get critical point using interval halving method
critical_point = Interval(J,a, b,n)
print("Critical point (minimum) using Interval Halving Method:", critical_point)
print("Value of J at critical point:", J(critical_point))


# (b) identify the critical point using Newton-Raphson method

# In[5]:


def J(w):
    return w**2 + (54/w)

def J_prime(w):
    return 2 * w - 54 / (w**2)

def J_double_prime(w):
    return 2 + 108 / (w**3)

def newton_raphson(initial_guess, tolerance=1e-8, max_iter=100):
    w = initial_guess
    for k in range(1, max_iter + 1):
        J_prime_w = J_prime(w)
        J_double_prime_w = J_double_prime(w)
        
        w_new = w - (J_prime_w / J_double_prime_w)
        
        if abs(J_prime_w) < tolerance:
            return w_new
        
        w = w_new
    
    return None

# Initial guess for Newton-Raphson method
initial_guess = 3

# Critical point using Newton-Raphson method
critical_point = newton_raphson(initial_guess)
print("Critical point (minimum) using Newton-Raphson Method:", critical_point)
print("Value of J at critical point:", J(critical_point))


#  (c) verify the result manually using the optimality criteria (post this write-up as well in .jpg/.png etc).

# # QUESTION-2

# Plot the surface J(w1, w2) = (w1 - 10)^2 + (w2 - 10)^2. Also, generated the corresponding contour plot. Label the plots appropriately. Give a suitable title for the figure.

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function J(w1, w2)
def J(w1, w2):
    return (w1 - 10)**2 + (w2 - 10)**2

# Generate data points for w1 and w2
w1 = np.linspace(0, 20, 100)
w2 = np.linspace(0, 20, 100)
w1, w2 = np.meshgrid(w1, w2)
J_values = J(w1, w2)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w1, w2, J_values, cmap='viridis')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('J(w1, w2)')
ax.set_title('Surface Plot of J(w1, w2)')
plt.show()

# Plot the contour
plt.figure(figsize=(8, 6))
contour = plt.contour(w1, w2, J_values, levels=20, cmap='viridis')
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Contour Plot of J(w1, w2)')
plt.colorbar(contour)
plt.show()


# # QUESTION-3

# Using line (unidirectional) search, for the function (w1 - 10 )^2 + (w2 - 10)^2, find the minimum value along the direction (2, 5). You can assume the start point to be (2,1).  Plot the function and its contours along with the minimum value in that direction.

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# Define the function J(w1, w2)
def J(w1, w2):
    return (w1 - 10)**2 + (w2 - 10)**2

# Define the starting point
w1_0, w2_0 = 2, 1

# Define the direction
direction = np.array([2, 5])

# Perform line search
step_sizes = np.linspace(0, 10, 100)
min_value = float('inf')
min_point = None
for step_size in step_sizes:
    new_point = np.array([w1_0, w2_0]) + step_size * direction
    new_value = J(*new_point)
    if new_value < min_value:
        min_value = new_value
        min_point = new_point


print("Coordinates of the minimum point:", min_point)
print("Value at the minimum point:", min_value)

# Plot the function and its contours
w1 = np.linspace(0, 20, 100)
w2 = np.linspace(0, 20, 100)
w1, w2 = np.meshgrid(w1, w2)
J_values = J(w1, w2)

plt.figure(figsize=(10, 8))
contour = plt.contour(w1, w2, J_values, levels=20, cmap='viridis')
plt.plot(min_point[0], min_point[1], 'ro')  # Plot minimum point
plt.text(min_point[0] + 0.5, min_point[1] + 0.5, f'Minimum: ({min_point[0]:.2f}, {min_point[1]:.2f})', color='red')
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Contour Plot of J(w1, w2) with Minimum Value along Direction (2, 5)')
plt.colorbar(contour)
plt.show()


# In[ ]:




