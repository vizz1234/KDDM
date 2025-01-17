import numpy as np

from sklearn.metrics.pairwise import euclidean_distances




def get_noise_dbscan(X, eps=0.0, min_samples=0):
    
    core_point_indices, noise_point_indices = None, None
    
    #########################################################################################
    ### Your code starts here ###############################################################
    
    ### 2.1 a) Identify the indices of all core points
    distances = euclidean_distances(X, X)
    neighbors = np.sum(distances <= eps, axis = 1)
    core_point_indices = np.argwhere(neighbors >= min_samples)
    core_point_indices = [x[0] for x in core_point_indices]
    ### Your code ends here #################################################################
    #########################################################################################
    
    
    #########################################################################################
    ### Your code starts here ###############################################################
    
    ### 2.1 b) Identify the indices of all noise points ==> noise_point_indices
    ind = np.argwhere(distances <= eps)
    mask = np.isin(ind[:, 0], core_point_indices, invert=True) & np.isin(ind[:, 1], core_point_indices)
    border_point_indices = ind[mask][:, 0]
    noise_mask = np.isin(np.arange(len(X)), core_point_indices, invert = True) & np.isin(np.arange(len(X)), border_point_indices, invert = True)
    noise_point_indices = np.arange(len(X))[noise_mask]
    # noise_point_indices = [list(x)[0] for x in noise_point_indices]

    ### Your code ends here #################################################################
    #########################################################################################
    
    return core_point_indices, noise_point_indices