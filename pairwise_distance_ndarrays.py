from scipy.spatial import distance
X = np.array([[1,2,3],[3,4,5],[6,7,8]])
Y = np.array([[1, 2, 3], [4, 5, 6]])
print(pairwise_distance_matrix(X,Y))
                         
def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)
    
    Returns
    --------
    D: matrix of shape (N, M), each entry D[i,j] is the distance between
    X[i,:] and Y[j,:] using the dot product.
    """
    N, D = X.shape
    M, _ = Y.shape
    distance_matrix = np.zeros((N, M), dtype=np.float)
    distance_matrix = distance.cdist(X, Y, 'euclidean')
    return distance_matrix
