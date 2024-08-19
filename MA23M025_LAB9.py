#!/usr/bin/env python
# coding: utf-8

# # QUESTION-2

# Using steepest gradient descent, find all the local minima for the function J(x1, x2) = (x1^2+x2−11)^2+(x1+x2^2−7)^2.
# 
# While applying gradient descent, do the following
# 
# (a) Fixing the value for alpha
# 
# (b) use line search to determine the value for alpha. Plot the intermediate steps in the iteration to show one of the minimal point.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Define the objective function J(x1, x2)
def objective_function(x1, x2):
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Define the gradient of J(x1, x2)
def gradient_function(x1, x2):
    df_dx1 = 4 * x1 * (x1**2 + x2 - 11) + 2 *(x1 + x2**2 - 7)
    df_dx2 = 2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7)
    return np.array([df_dx1, df_dx2])

# Perform gradient descent
def gradient_descent(starting_point, alpha=0.01, max_iterations=1000, tolerance=1e-8):
    x = starting_point
    for _ in range(max_iterations):
        grad = gradient_function(*x)
        x_new = x - alpha * grad
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x

# Set starting points for gradient descent
starting_points = [
    np.array([-4.0, -4.0]),
    np.array([-3.0, 4.0]),
    np.array([3.0, -2.0]),
    np.array([3.0, 3.0])
]

# Perform gradient descent for each starting point and plot the optimization path
for i, starting_point in enumerate(starting_points):
    minima = gradient_descent(starting_point)
    print(f"Local minimum (Starting Point {i+1}):", minima)
    print("Value of J at local minimum:", objective_function(*minima))

    # Plot the optimization path
    path = [starting_point]
    x = starting_point
    for _ in range(1000):
        grad = gradient_function(*x)
        x_new = x - 0.01 * grad
        path.append(x_new)
        if np.linalg.norm(x_new - x) < 1e-8:
            break
        x = x_new
    path = np.array(path)
        
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = objective_function(X1, X2)

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=np.logspace(0, 5, 35), cmap='viridis', alpha=0.8)
    plt.colorbar(label='Objective Function Value')
    plt.plot(path[:, 0], path[:, 1], '-o', color='r', markersize=5, label='Optimization Path')
    plt.plot(minima[0], minima[1], 'go', markersize=8, label='Local Minimum')
    plt.plot(starting_point[0], starting_point[1], 'bo', markersize=8, label='Starting Point')
    plt.title(f'Gradient Descent Optimization Path - Starting Point {i+1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import math as m

def grad_J(w):   # calculating gradient of the function at a point.
    length = np.size(w)
    grad = np.zeros(length)
    
    grad[0] = (4*(w[0]**3)) - (42* w[0]) + (4*w[0]*w[1]) + (2 * (w[1]**2)) - 14
    grad[1] = (2*(w[0]**2)) + (4*w[0]*w[1]) - (26 * w[1]) + (4 * (w[1]**3)) - 22
    
    return grad

def phi(w):
    val = ((w[0]**2) + w[1] - 11)**2 + (w[0] + (w[1]**2) - 7)**2
    return val


def bracketing_method_alpha(w_start, direction, a, b, n):
    d_alpha = (b - a)/n
    w1 = a
    w2 = w1 + d_alpha
    w3 = w2 + d_alpha
    while(w3 <= b):
        x = w_start + w1*direction
        y = w_start + w2*direction
        z = w_start + w3*direction
        if((phi(x) >= phi(y)) and (phi(y) <= phi(z))):
            return w1, w3
        else:
            w1 = w2
            w2 = w3
            w3 = w3 + d_alpha
    if(phi(w_start + a*direction) > phi(w_start + b*direction)):
        return (b - d_alpha), b
    else:
        return a, (a + d_alpha)

def Region_elimination_alpha(w_start, direction, a, b):
    L = b - a
    eps = 1e-5
    w_m = (a+b)/2
    while(L > eps):
        w_1 = a + L/4
        w_2 = b - L/4
        x = w_start + w_1*direction
        y = w_start + w_m*direction
        z = w_start + w_2*direction
        if(phi(x) < phi(y)):
            b = w_m
            w_m = w_1
        elif(phi(z) < phi(y)):
            a = w_m
            w_m = w_2
        else:
            a = w_1
            b = w_2   
        L = b - a
    return a
        

def Unidirectional(w_start , direction, n):
    alpha_1 = 0
    alpha_2 = 5
    interval = bracketing_method_alpha(w_start, direction, alpha_1, alpha_2, n)
    
    pt = Region_elimination_alpha(w_start, direction, interval[0], interval[1])
    return pt
    
def Gradient_descent(st_point):    # Using unidirectional method to find learning rate.
    w = st_point
    grad = grad_J(w)
    eps = 10e-5
    iterates = []
    iterates.append(list(w))
    
    while(pow(np.dot(grad, grad), 0.5) > eps):
        direction = - grad
        alpha = Unidirectional(w, direction, 50)
        w = w - alpha*grad
        iterates.append(list(w))
        grad = grad_J(w)
        
    iterates = np.array(iterates)    
    return w, iterates



starting_point = np.array([-1, 0])
optimal_point , iteration_point = Gradient_descent(starting_point)
print(f'The optimal point is {optimal_point}')
print('The iteration points are')
print(iteration_point)


w1 = np.arange(-5, 5, 0.1)
w2 = np.arange(-5, 5, 0.1)
W1, W2 = np.meshgrid(w1, w2)  #Forming MeshGrid

J = ((W1**2) + W2 - 11)**2 + (W1 + (W2**2) - 7)**2

fig, ax = plt.subplots(1, 1) 
ax.contour(W1, W2, J, 30)
ax.plot(iteration_point[:, 0], iteration_point[:, 1], "X-")
ax.set_xlabel('w1 Label')
ax.set_ylabel('w2 Label')
ax.set_title('Using gradient descent algorithm where learning rate is computed using unidirectional method')
plt.show()



# In[3]:


starting_point = np.array([-4, 3])
optimal_point , iteration_point = Gradient_descent(starting_point)
print(f'The optimal point is {optimal_point}')
print('The iteration points are')
print(iteration_point)


w1 = np.arange(-5, 5, 0.1)
w2 = np.arange(-5, 5, 0.1)
W1, W2 = np.meshgrid(w1, w2)  #Forming MeshGrid

J = ((W1**2) + W2 - 11)**2 + (W1 + (W2**2) - 7)**2

fig, ax = plt.subplots(1, 1) 
ax.contour(W1, W2, J, 30)
ax.plot(iteration_point[:, 0], iteration_point[:, 1], "X-")
ax.set_xlabel('w1 Label')
ax.set_ylabel('w2 Label')
ax.set_title('Using gradient descent algorithm where learning rate is computed using unidirectional method')
plt.show()


# In[4]:


starting_point = np.array([-3, 2])
optimal_point , iteration_point = Gradient_descent(starting_point)
print(f'The optimal point is {optimal_point}')
print('The iteration points are')
print(iteration_point)


w1 = np.arange(-5, 5, 0.1)
w2 = np.arange(-5, 5, 0.1)
W1, W2 = np.meshgrid(w1, w2)  #Forming MeshGrid

J = ((W1**2) + W2 - 11)**2 + (W1 + (W2**2) - 7)**2

fig, ax = plt.subplots(1, 1) 
ax.contour(W1, W2, J, 30)
ax.plot(iteration_point[:, 0], iteration_point[:, 1], "X-")
ax.set_xlabel('w1 Label')
ax.set_ylabel('w2 Label')
ax.set_title('Using gradient descent algorithm where learning rate is computed using unidirectional method')
plt.show()


# In[5]:


starting_point = np.array([4, 4])
optimal_point , iteration_point = Gradient_descent(starting_point)
print(f'The optimal point is {optimal_point}')
print('The iteration points are')
print(iteration_point)


w1 = np.arange(-5, 5, 0.1)
w2 = np.arange(-5, 5, 0.1)
W1, W2 = np.meshgrid(w1, w2)  #Forming MeshGrid

J = ((W1**2) + W2 - 11)**2 + (W1 + (W2**2) - 7)**2

fig, ax = plt.subplots(1, 1) 
ax.contour(W1, W2, J, 30)
ax.plot(iteration_point[:, 0], iteration_point[:, 1], "X-")
ax.set_xlabel('w1 Label')
ax.set_ylabel('w2 Label')
ax.set_title('Using gradient descent algorithm where learning rate is computed using unidirectional method')
plt.show()


# In[ ]:





# In[ ]:




