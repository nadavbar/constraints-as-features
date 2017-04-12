import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
from sklearn.manifold import SpectralEmbedding


def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return n*j - j*(j+1)/2 + i - 1 - j

def handle_must_link_constraints(d, must_link_constraints, n):
    d_transform = d.copy()
    min_dist = d_transform.min()
    # update must link with minimal distance
    for c in must_link_constraints:
        i,j = c
        d_transform[square_to_condensed(i,j, n)] = min_dist

        for x in range(1, n):
            for y in range(x):
                
                id_xy = square_to_condensed(x,y,n)
                d_xy = d_transform[id_xy]
                if x == i or j ==y:
                    d_xi_jy = np.inf
                else:
                    d_xi_jy = d_transform[square_to_condensed(x,i,n)] + d_transform[square_to_condensed(j,y,n)]

                if x == j or y == i:
                    d_xj_iy = np.inf
                else:
                    d_xj_iy = d_transform[square_to_condensed(x,j,n)] + d_transform[square_to_condensed(i,y,n)]
                
                d_transform[id_xy] = min(d_xy, d_xi_jy, d_xj_iy)

    return d_transform

def handle_cannot_link_constraints(X, d_transform, cannot_link_constraints, n, norm_p, spectral_embedding_components=None, sc_sigma=1):
    alpha = d_transform.max()
    d_p = np.power(d_transform, norm_p)

    if spectral_embedding_components is None:
        spectral_embedding_components = len(X[0])
    
    sc_embedding = SpectralEmbedding(n_components=spectral_embedding_components, affinity="precomputed")
    affinity_mat = np.exp(- np.power(squareform(d_transform), 2) / (2 *(sc_sigma ** 2)))
    embedding = sc_embedding.fit_transform(affinity_mat)

    for c in cannot_link_constraints:
        i,j = c
        # make sure that alwas i > j
        if i < j:
            i, j = j, i


        e_i = embedding[i]
        e_j = embedding[j]

        for x in range(1, n):
            for y in range(x):
                
                if x == i and y == j:
                    val = 2
                else:
                    e_x = embedding[x]
                    e_y = embedding[y]

                    # use l2 norm distance here.
                    d_ex_ej = norm(e_x - e_j)
                    d_ex_ei =  norm(e_x-e_i)
                    v_x = (d_ex_ej - d_ex_ei) / (d_ex_ej + d_ex_ei)

                    d_ey_ej = norm(e_y - e_j)
                    d_ey_ei = norm(e_y - e_i)
                    v_y = (d_ey_ej - d_ey_ei) / (d_ey_ej + d_ey_ei)

                    val = np.abs(v_x - v_y)
                    
                d_p[square_to_condensed(x, y, n)] += np.power(val * alpha, norm_p)
    
    return np.power(d_p, 1 / float(norm_p))

class ConstraintsFeaturesTransform():

    def __init__(self, norm_p=2, spectral_embedding_components=None):
        self.norm_p = norm_p
        self.spectral_embedding_components = spectral_embedding_components

    # X - data matrix, each row is a different observation ("data point")
    # must_link_constraints - A matrix/ndarray in the shape of m X 2, where m is the number of 
    # must link contraints.
    # each row is in the form of [first point index, second point index]
    # cannot_link_constraints - A matrix/ndarray in the shape of k x 2, where k is the number of cannot link
    # constraints. Each row is in the form of [first point index, second point index]
    # p - the power of the Frobenius / Euclidean norm
    def transform(self, X, must_link_constraints, cannot_link_constraints,):
        d_original = pdist(X, p=self.norm_p)
        n = len(X)
        if must_link_constraints is not None:
            d_transform = handle_must_link_constraints(d_original, must_link_constraints, n)
        else:
            d_transform = d_original

        if cannot_link_constraints is not None:
            d_transform = handle_cannot_link_constraints(X, d_transform, cannot_link_constraints, n, self.norm_p, self.spectral_embedding_components)

        return d_transform