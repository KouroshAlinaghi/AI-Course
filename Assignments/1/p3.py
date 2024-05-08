import numpy as np
import tensorflow as tf

def smote(X, y, k=5):
    """
    SMOTE function to generate synthetic minority class instances
    :param X: input features
    :param y: target variable
    :param k: number of nearest neighbors to consider
    :return: X_resampled, y_resampled
    """
    # Find the indices of minority class instances
    idx_minority = np.where(y == -1)[0]
    
    # Calculate the distances between minority instances and their neighbors
    dists = np.zeros((len(idx_minority), len(idx_minority)))
    for i in range(len(idx_minority)):
        for j in range(len(idx_minority)):
            if i != j:
                dists[i][j] = np.linalg.norm(X[idx_minority[i]] - X[idx_minority[j]])
                
    # Sort the distances in ascending order
    sorted_dists = np.argsort(dists, axis=0)
    
    # Select the nearest neighbors for each minority instance
    neighbors = []
    for i in range(len(idx_minority)):
        for j in range(k):
            neighbors.append(idx_minority[sorted_dists[i][j]])
    
    # Generate synthetic instances between minority instances and their neighbors
    X_resampled = []
    y_resampled = []
    for i in range(len(idx_minority)):
        if len(neighbors) > 0:
            idx_neighbor = np.random.choice(neighbors)
            neighbor = X[idx_neighbor]
            alpha = np.random.uniform(low=0, high=1)
            X_resampled.append(neighbor + alpha * (neighbor - X[idx_minority[i]]))
            y_resampled.append(y[idx_minority[i]])
            neighbors.remove(idx_neighbor)
    
    return X_resampled, y_resampled
