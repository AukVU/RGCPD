import os
import scipy
from scipy import random
from scipy import sparse
from scipy.sparse.linalg import eigs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Plots in 3D
from c_functions import create_graph

from typing import Union


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def check_stability(graph: Union[np.ndarray, dict], lag_first_axis: bool = False, verbose: bool = False):
    """
    Raises an AssertionError if the input graph corresponds to a non-stationary
    process.
    Parameters
    ----------
    graph: array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)
    lag_first_axis: bool
        Indicates if the lag is in the first axis or in the last
    verbose: bool
        Level of output information
    """

    if type(graph) == dict:
        graph = create_graph(graph, return_lag=False)

    # Adapt the Varmodel return to the desired format (lag, N, N) -> (N, N, lag)
    if lag_first_axis:
        graph = np.moveaxis(graph, 0, 2)

    if verbose:
        print("The shape of the graph is", graph.shape)

    # Get the shape from the input graph
    n_nodes, _, period = graph.shape
    # Set the top section as the horizontally stacked matrix of
    # shape (n_nodes, n_nodes * period)
    stability_matrix = \
        scipy.sparse.hstack([scipy.sparse.lil_matrix(graph[:, :, t_slice])
                             for t_slice in range(period)])
    # Extend an identity matrix of shape
    # (n_nodes * (period - 1), n_nodes * (period - 1)) to shape
    # (n_nodes * (period - 1), n_nodes * period) and stack the top section on
    # top to make the stability matrix of shape
    # (n_nodes * period, n_nodes * period)
    stability_matrix = \
        scipy.sparse.vstack([stability_matrix,
                             scipy.sparse.eye(n_nodes * (period - 1),
                                              n_nodes * period)])
    # Check the number of dimensions to see if we can afford to use a dense
    # matrix
    n_eigs = stability_matrix.shape[0]
    if n_eigs <= 25:
        # If it is relatively low in dimensionality, use a dense array
        stability_matrix = stability_matrix.todense()
        eigen_values, _ = scipy.linalg.eig(stability_matrix)
    else:
        # If it is a large dimensionality, convert to a compressed row sorted
        # matrix, as it may be easier for the linear algebra package
        stability_matrix = stability_matrix.tocsr()
        # Get the eigen values of the stability matrix
        eigen_values = scipy.sparse.linalg.eigs(stability_matrix,
                                                k=(n_eigs - 2),
                                                return_eigenvectors=False)
    # Ensure they all have less than one magnitude

    assert np.all(np.abs(eigen_values) < 1.), \
        "Values given by time lagged connectivity matrix corresponds to a " + \
        " non-stationary process!"

    if verbose:
        print("The coefficients correspond to an stationary process")


def create_random_mode(size: tuple, mu: tuple = (0, 0), var: tuple = (.5, .5),
                       position: tuple = (3, 3, 3, 3), plot: bool = False,
                       Sigma: np.ndarray = None, random: bool = True) -> np.ndarray:
    """
    Creates a positive-semidefinite matrix to be used as a covariance matrix of two var
    Then use that covariance to compute a pdf of a bivariate gaussian distribution which
    is used as mode weight. It is random but enfoced to be spred.
    Inspired in:  https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    and https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices

    :param random: Does not create a random, insted uses a ind cov matrix
    :param size
    :param mu tuple with the x and y mean
    :param var used to enforce spread modes. (0, 0) = totally random
    :param position: tuple of the position of the mean
    :param plot:
    """

    # Unpack variables
    size_x, size_y = size
    x_a, x_b, y_a, y_b = position
    mu_x, mu_y = mu
    var_x, var_y = var

    # In case of non invertible
    if Sigma is not None:
        Sigma_o = Sigma.copy()
    else:
        Sigma_o = Sigma

    # Compute the position of the mean
    X = np.linspace(-x_a, x_b, size_x)
    Y = np.linspace(-y_a, y_b, size_y)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Mean vector
    mu = np.array([mu_x, mu_y])

    # Compute almost-random covariance matrix
    if random:
        Sigma = np.random.rand(2, 2)
        Sigma = np.dot(Sigma, Sigma.transpose())  # Make it invertible
        Sigma += + np.array([[var_x, 0], [0, var_y]])
    else:
        if Sigma is None:
            Sigma = np.asarray([[0.5, 0], [0, 0.5]])

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    # The actual weight
    Z = np.exp(-fac / 2) / N

    if not np.isfinite(Z).all() or (Z > 0.5).any():
        Z = create_random_mode(size=size, mu=mu, var=var, position=position,
                               plot=False, Sigma=Sigma_o, random=random)

    if plot:
        # Create a surface plot and projected filled contour plot under it.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                        cmap=cm.viridis)

        cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

        # Adjust the limits, ticks and view angle
        ax.set_zlim(-0.15, 0.2)
        ax.set_zticks(np.linspace(0, 0.2, 5))
        ax.view_init(27, -21)
        plt.show()
        plt.close()

    return Z

# def create_cov_matrix(noise_weights, spatial_covariance=0.4, use_spataial_cov=True):
#     #TODO: needs to be fixed, because covariance matrix does not work with random relations.
#     """
#     Use spatial covariance, no acaba d'anar...
#     :param noise_weights:
#     :param spatial_covariance:
#     :param use_spataial_cov:
#     :return:
#     """
#     grid_points = np.prod(noise_weights.shape[1:])
#     cov = np.zeros((grid_points, grid_points))  # noise covariance matrix
#     for n in noise_weights:
#         flat_n = n.reshape(grid_points)
#         nonzero = n.reshape(grid_points).nonzero()[0]
#         for i in nonzero:
#             if use_spataial_cov:
#                 cov[i, nonzero] = spatial_covariance
#             else:
#                 cov[i, nonzero] = flat_n[nonzero] * spatial_covariance
#
#     np.fill_diagonal(cov, 1)  # Set diagonal to 1
#
#     assert np.all(np.linalg.eigvals(cov) > 0), "covariance matrix not positive-semidefinte"
#
#     return cov


