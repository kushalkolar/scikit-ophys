from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import randomized_svd
import jax
import jax.numpy as jnp


@dataclass
class InitArrays:
    Hw: np.ndarray = None    # whitened vstack([P, F])

    P: np.ndarray  = None # past lag vectors centered, not whitened
    F: np.ndarray  = None # future lag vectors centered, not whitened

    Pw: np.ndarray  = None # past lag vectors, whitened
    Fw: np.ndarray  = None # future lag vectors, whitened

    Z: np.ndarray = None   # whitening matrix

    Zp: np.ndarray = None  # whitening matrix for past
    Zf: np.ndarray = None  # whitening matrix for future

    mu_p: np.ndarray  = None # mean of the past, shape is [n_pixels]
    mu_f: np.ndarray  = None # mean of the future, shape is [n_pixels]

    Y: np.ndarray  = None    # output, shape is [k, n_timepoints]

    W: np.ndarray  = None     # W matrix
    W_nw: np.ndarray  = None # W matrix with non-whitened data projected onto Y
    M: np.ndarray  = None    # M matrix

    k: int  = None           # number of components
    lag: int  = None
    lag_step: int = None


@jax.jit
def sm_update_step(Y, X, eta):
    """jax jitted similarity matching update step"""
    Y_old = Y

    Y_new = Y + eta * Y @ (X - Y.T @ Y)

    Y_new = jnp.clip(Y_new, 0)

    error = jnp.divide(jnp.abs(Y_new - Y_old), jnp.abs(Y_old + 1e-2)).max() / eta

    return Y_new, error


def nnsvd(A, k, eps: float = 1e-6):
    U, S, V = randomized_svd(A, n_components=k, n_iter=15)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, k):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = np.linalg.norm(x_p, ord=2), np.linalg.norm(y_p, ord=2)
        x_n_nrm, y_n_nrm = np.linalg.norm(x_n, ord=2), np.linalg.norm(y_n, ord=2)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    return W, H


def estimate_n_components_kmeans(
        H_nw: np.ndarray,
        max_k: int,
        max_iters=100,
        tol=1e-4
    ) -> int:
    """
    Estimate number of components using 1d kmeans cluster

    Parameters
    ----------
    H_nw: Non-whitened H matrix, i.e. vstack([past, future])

    max_k: int, upper bound on k estimate

    """

    # _, s_vals, _ = np.linalg.svd(H_nw)

    print("running svd to estimate k")
    _, s_vals, _ = svds(H_nw, k=max_k + 10)

    s_vals = s_vals[::-1]  # svds returns in reverse order

    s_vals = np.log(s_vals[1:max_k + 1] / s_vals.max())

    # Randomly initialize k cluster centers
    centroids = np.random.choice(s_vals, 2 , replace=False)
    centroids[0] = np.max(s_vals)
    for entry in range(1, 2):
        centroids[entry] = np.min(s_vals)

    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        distances = np.abs(s_vals[:, np.newaxis] - centroids[np.newaxis, :])
        labels = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array(
            [s_vals[labels == i].mean() if np.any(labels == i) else centroids[i] for i in range(2)]
        )

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    n_components = (labels == 0).sum() + 1

    return n_components


def eigen_decomposition(M):
    # output eigenvalues and eigenvectors sorted by eigenvalues

    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors


def truncated_whitening(X, X_mc, k) -> tuple[np.ndarray, np.ndarray]:
    # we assume X has a shape (N,T)
    # N = number of features
    # T = number of samples

    cov = (X_mc @ X_mc.T)
    cov = cov / X_mc.shape[1] #np.linalg.norm(cov, ord="fro")
    S, U = eigen_decomposition(cov)

    S_eco = S[:k]
    U_eco = U[:, :k]

    # Epsilon is added to the eigenvalues to prevent division by zero
    epsilon = 1e-5
    inv_sqrt_S = np.diag(1.0 / np.sqrt(S_eco + epsilon))

    # whitening matrix W shape (N, N)
    W = U_eco @ inv_sqrt_S @ U_eco.T

    # whitened data shape (N, T)
    Xw = W @ X

    return Xw, W


def get_permute_indices(A):
    # Convert maximization to minimization by negating the matrix
    cost_matrix = -A
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return col_indices
