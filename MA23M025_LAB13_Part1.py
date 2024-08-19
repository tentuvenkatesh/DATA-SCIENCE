#!/usr/bin/env python
# coding: utf-8

# # QUESTION 1:

# Create the following data and write to a csv file: Generate 10 random points in each of the the following circles
# 
# (i) centre at (3,3) and radius 2, 
# 
# (ii) centre at (7,7) and radius 2 
# 
# (iii) centre at (11,11) and radius 2.  
# 
# Plot the data as well. 

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(center, radius, num_points):
    points = []
    for i in range(num_points):
        r = np.random.uniform(0, radius)
        theta = np.random.uniform(0, 2*np.pi)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points

# Define circle parameters
circles = [
    {'center': (3, 3), 'radius': 2, 'num_points': 10, 'label': 'Circle 1'},
    {'center': (7, 7), 'radius': 2, 'num_points': 10, 'label': 'Circle 2'},
    {'center': (11, 11), 'radius': 2, 'num_points': 10, 'label': 'Circle 3'}
]

# Generate and combine random points for all circles
all_points = []
for circle in circles:
    center = circle['center']
    radius = circle['radius']
    num_points = circle['num_points']
    points = generate_random_points(center, radius, num_points)
    all_points.extend(points)

# Convert points to DataFrame
df_all = pd.DataFrame(all_points, columns=['x', 'y'])

# Plot all circles and points
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the circles
for circle in circles:
    center = circle['center']
    radius = circle['radius']
    label = circle['label']
    circle_plot = plt.Circle(center, radius, color='red', fill=False, label=label)
    ax.add_artist(circle_plot)

# Plot the points
plt.scatter(df_all['x'], df_all['y'], color='blue', label='All Points')

# Set plot limits based on circles
max_radius = max(circle['radius'] for circle in circles)
min_x = min(circle['center'][0] - max_radius for circle in circles)
max_x = max(circle['center'][0] + max_radius for circle in circles)
min_y = min(circle['center'][1] - max_radius for circle in circles)
max_y = max(circle['center'][1] + max_radius for circle in circles)
ax.set_xlim(min_x - 1, max_x + 1)
ax.set_ylim(min_y - 1, max_y + 1)

# Plot settings
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Points in Circles')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True)
plt.show()


# # QUESTION-2:

# Implement K - means clustering algorithm and for the above data, show the change in the centroid as well as the class assignments. Also, plot the cost function for K varying from 1 to 5. Show that the value of K matches with the intuition from the data. Plot the K-classes for the final K-value.

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(center, radius, num_points):
    points = []
    for i in range(num_points):
        r = np.random.uniform(0, radius)
        theta = np.random.uniform(0, 2*np.pi)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points

# Define circle parameters
circles = [
    {'center': (3, 3), 'radius': 2, 'num_points': 10, 'label': 'Circle 1'},
    {'center': (7, 7), 'radius': 2, 'num_points': 10, 'label': 'Circle 2'},
    {'center': (11, 11), 'radius': 2, 'num_points': 10, 'label': 'Circle 3'}
]

# Generate and combine random points for all circles
all_points = []
for circle in circles:
    center = circle['center']
    radius = circle['radius']
    num_points = circle['num_points']
    points = generate_random_points(center, radius, num_points)
    all_points.extend(points)

# Convert points to DataFrame
df_all = pd.DataFrame(all_points, columns=['x', 'y'])

# Save all points to CSV
df_all.to_csv('all_points.csv', index=False)

# Print the CSV file contents
print("Contents of all_points.csv:")
print(df_all)


# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('all_points.csv')

# Input K (number of classes)
K_values = [1, 2, 3, 4, 5]
costs = []

# Perform K-means clustering for each K
for K in K_values:
    # Initialize k centroid locations randomly
    centroids = df.sample(K, random_state=42)

    # Perform K-means clustering
    max_iter = 100
    cost_history = []
    for _ in range(max_iter):
        # Compute distances from each point to each centroid
        distances = np.linalg.norm(df[['x', 'y']].values[:, :, np.newaxis] - centroids[['x', 'y']].values.T[np.newaxis, :, :], axis=1)
        
        # Assign cluster index (index of the closest centroid) to each data point
        cluster_indices = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([df[cluster_indices == k][['x', 'y']].mean() for k in range(K)])
        
        # Calculate cost function
        cost = np.sum(np.min(distances, axis=1)) / len(df)
        cost_history.append(cost)
        
        # Check for convergence
        if np.allclose(centroids.values, new_centroids):
            break
        
        centroids = pd.DataFrame(new_centroids, columns=['x', 'y'])
    
    costs.append(cost_history)

# Plot the changes in centroid locations over iterations for each K
for i, K in enumerate(K_values):
    centroid_history = [df.sample(K, random_state=42)]  # Initialize with random centroids
    centroids = centroid_history[0]

    for j in range(max_iter):
        distances = np.linalg.norm(df[['x', 'y']].values[:, :, np.newaxis] - centroids[['x', 'y']].values.T[np.newaxis, :, :], axis=1)
        cluster_indices = np.argmin(distances, axis=1)
        new_centroids = np.array([df[cluster_indices == k][['x', 'y']].mean() for k in range(K)])

        if np.allclose(centroids.values, new_centroids):
            break

        centroids = pd.DataFrame(new_centroids, columns=['x', 'y'])
        centroid_history.append(centroids.copy())

    # Plot centroid trajectories and cost function
    num_plots = len(centroid_history)
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5), sharex=True, sharey=True)

    for j, centroids in enumerate(centroid_history):
        ax = axes[j]
        ax.scatter(df['x'], df['y'], label='Data Points')
        ax.scatter(centroids['x'], centroids['y'], color='red', label='Centroids')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Iteration {j+1}')
        ax.legend()

    plt.suptitle(f'Centroid Locations over Iterations for K = {K}')
    plt.tight_layout()
    plt.show()


# In[48]:


# Compute costs for the Elbow Method
costs = [cost[-1] for cost in costs]

# Plot the costs
plt.figure(figsize=(8, 5))
plt.plot(K_values, costs, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost')
plt.title('Elbow Method for Optimal k')
plt.xticks(K_values)
plt.grid(True)
plt.show()

