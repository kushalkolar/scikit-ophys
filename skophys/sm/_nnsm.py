from tqdm import tqdm
import numpy as np
import scipy
from jax import numpy as jnp
from qpsolvers import solve_qp


from ..preprocessing import UnVectorizer, Vectorizer
from ._base import BaseSM
from .utils import InitArrays, estimate_n_components_kmeans, truncated_whitening, nnsvd, sm_update_step


class SM(BaseSM):
    def __init__(
        self,
        lag: int,
        lag_step: int,
    ):
        self._lag: int = lag
        self._lag_step: int = lag_step
        self._lag_slices: list[slice] = None

        self._movie: np.ndarray = None
        self._X_init: np.ndarray = None
        self._unvec: UnVectorizer = None

        self._n_pixels: int = None

        self._cov_init: np.ndarray = None

        self._pre_init_arrays: InitArrays = None
        self._init_arrays: InitArrays = None

    def fit(
        self,
        data: np.ndarray,
        max_k: int,
        k: int = None,
        k_add_trunc_w: int = 0,
        method: str = "nnsvd"
    ):
        """
        Run nnSVD to pre-initialize Y before initializing with gradient descent.

        Parameters
        ----------
        data: data to initialize with, shape of [n_timepoints, rows, cols] or [n_pixels, n_timepoints]

        method: only "nnsvd" for now

        k_add_trunc_w: int

            k estimated using k-means clustering of singular values of centered data.
            if `k_add_trunc_w` is specific, this value is added to the estimated k
            which specifies the rank of the truncated data used to produce the whitening matrix

        """
        if data.ndim == 3:
            self._movie = data
            self._X_init = Vectorizer().transform(self.movie)
            self._unvec = UnVectorizer(self.movie.shape[1:])
        elif data.ndim == 2:
            self._X_init = data
        else:
            raise ValueError("`data` ndim must be 2 or 3")

        # mean center
        self._X_init = self.X_init - self.X_init.mean(axis=1)[:, None]

        self._n_pixels = self._X_init.shape[0]

        if method != "nnsvd":
            raise ValueError("only nnsvd supported for now")

        Hs = list()

        print("creating H")
        for i in range(self.n_pixels):
            Hs.append(
                scipy.linalg.hankel(
                    self.X_init[i, : self.lag * self.lag_step * 2],
                    self.X_init[i, (self.lag * self.lag_step * 2) - 1 :],
                )[:: self.lag_step]
            )

        H = np.zeros(
            (
                self.lag * 2 * self.n_pixels,
                self.X_init.shape[1] - (self.lag * self.lag_step * 2) + 1,
            )
        )

        for i, H_i in enumerate(Hs):
            H[i :: self.n_pixels] = H_i

        past = H[: self.lag * self.n_pixels].copy()

        mu0 = H[: self.lag * self.n_pixels].mean(axis=1)

        H[: self.lag * self.n_pixels] -= mu0[:, None]

        past_mc = H[: self.lag * self.n_pixels].copy()

        print("computing covariance")
        # cov = past.T @ past

        # self._cov_init = cov / np.linalg.norm(cov, ord="fro")

        if k is None:
            print("estimating k")
            k = estimate_n_components_kmeans(
                H_nw=past_mc,
                max_k=max_k,
            )

        print("truncated whitening")

        past_w, Z = truncated_whitening(past, past_mc, k=k + k_add_trunc_w)

        self._cov_init = (past_w.T @ past_w) / past_w.shape[1]

        self.mu = mu0
        self.past = past
        self.past_mc = past_mc
        self.past_w = past_w
        self.Z = Z
        self.k = k

        print("performing SVD")
        return np.linalg.svd(self.cov_init, full_matrices=False)


class NNSM(BaseSM):
    def __init__(
        self,
        lag: int,
        lag_step: int,
    ):
        self._lag: int = lag
        self._lag_step: int = lag_step
        self._lag_slices: list[slice] = None

        self._movie: np.ndarray = None
        self._X_init: np.ndarray = None
        self._unvec: UnVectorizer = None

        self._n_pixels: int = None

        self._cov_init: np.ndarray = None

        self._pre_init_arrays: InitArrays = None
        self._init_arrays: InitArrays = None

    def pre_initialize(
        self,
        data: np.ndarray,
        max_k: int,
        k: int = None,
        k_add_trunc_w: int = 0,
        method: str = "nnsvd"
    ):
        """
        Run nnSVD to pre-initialize Y before initializing with gradient descent.

        Parameters
        ----------
        data: data to initialize with, shape of [n_timepoints, rows, cols] or [n_pixels, n_timepoints]

        method: only "nnsvd" for now

        k_add_trunc_w: int

            k estimated using k-means clustering of singular values of centered data.
            if `k_add_trunc_w` is specific, this value is added to the estimated k
            which specifies the rank of the truncated data used to produce the whitening matrix

        """
        if data.ndim == 3:
            self._movie = data
            self._X_init = Vectorizer().transform(self.movie)
            self._unvec = UnVectorizer(self.movie.shape[1:])
        elif data.ndim == 2:
            self._X_init = data
        else:
            raise ValueError("`data` ndim must be 2 or 3")

        # mean center
        self._X_init = self.X_init - self.X_init.mean(axis=1)[:, None]

        self._n_pixels = self._X_init.shape[0]

        if method != "nnsvd":
            raise ValueError("only nnsvd supported for now")

        Hs = list()

        print("creating H")
        for i in range(self.n_pixels):
            Hs.append(
                scipy.linalg.hankel(
                    self.X_init[i, : self.lag * self.lag_step * 2],
                    self.X_init[i, (self.lag * self.lag_step * 2) - 1 :],
                )[:: self.lag_step]
            )

        H = np.zeros(
            (
                self.lag * 2 * self.n_pixels,
                self.X_init.shape[1] - (self.lag * self.lag_step * 2) + 1,
            )
        )

        for i, H_i in enumerate(Hs):
            H[i :: self.n_pixels] = H_i

        past = H[: self.lag * self.n_pixels].copy()

        mu0 = H[: self.lag * self.n_pixels].mean(axis=1)

        H[: self.lag * self.n_pixels] -= mu0[:, None]

        past_mc = H[: self.lag * self.n_pixels].copy()

        print("computing covariance")
        # cov = past.T @ past

        # self._cov_init = cov / np.linalg.norm(cov, ord="fro")

        if k is None:
            print("estimating k")
            k = estimate_n_components_kmeans(
                H_nw=past_mc,
                max_k=max_k,
            )

        print("truncated whitening")

        past_w, Z = truncated_whitening(past, past_mc, k=k + k_add_trunc_w)

        self._cov_init = (past_w.T @ past_w) / past_w.shape[1]

        print("performing nnSVD")
        _, Y = nnsvd(A=self.cov_init, k=k)

        # Y /= Y.max()

        self._pre_init_arrays = InitArrays(
            Hw=None,
            P=past,
            F=None,
            Pw=past_w,
            Fw=None,
            mu_p=mu0,
            mu_f=None,
            Z=Z,
            Y=Y,
            k=k,
            lag=None,
            lag_step=None,
            W=None,
            W_nw=None,
            M=None,
        )

    def initialize(
        self,
        max_iter: int = 2_000,
        eta: float = 0.1,
        error_threshold: float = 1e-2,
        inertia: float = 1e-3,
        keep_iteration_results: bool = False,
    ):
        if self._pre_init_arrays is None:
            raise AttributeError("Must pre-initialize first")

        Y = self._pre_init_arrays.Y.copy()

        # T = self.cov_init.shape[0]

        error_log = list()

        Ys = list()
        Ws = list()

        for iteration in tqdm(range(max_iter)):
            Y, err = sm_update_step(
                jnp.array(Y),
                jnp.array(self.cov_init),
                jnp.array(eta, dtype=jnp.float32),
            )

            error_log.append(float(err))

            # below inertia threshold, error is decreasing and below error threshold
            if iteration > 2:
                if (error_log[-2] - error_log[-1]) < inertia and (
                    error_log[-1] < error_log[-2]
                ) and (error_log[-1] < error_threshold):
                    break

        Y = np.array(Y)

        W = Y @ self._pre_init_arrays.Pw.T  # / sum_y_squared
        M = Y @ Y.T  # / sum_y_squared

        self._init_arrays = InitArrays(
            Hw=None,
            P=None,
            F=None,
            Pw=None,
            Fw=None,
            mu_p=self._pre_init_arrays.mu_p,
            mu_f=None,
            Y=Y,
            k=self._pre_init_arrays.k,
            lag=None,
            lag_step=None,
            W=W,
            W_nw=None,
            M=M,
        )

        return error_log, Ys, Ws

    def fit_timepoint(self, x_t: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x_t: vector of shape [n_pixels], non-whitened, non-centered

        Returns
        -------
        y_t: vector of shape [k, ]

        """

        if x_t.ndim > 1:
            x_t = x_t.ravel()

        # whiten
        x_t = self.pre_init_arrays.Z @ x_t

        x_t = x_t.astype(np.float64)

        Wt = self.init_arrays.W.astype(np.float64)
        Mt = self.init_arrays.M.astype(np.float64)
        k = self.init_arrays.k

        return solve_qp((Mt + Mt.T) / 2, -Wt @ x_t, lb=np.zeros(k), solver="cvxopt")
