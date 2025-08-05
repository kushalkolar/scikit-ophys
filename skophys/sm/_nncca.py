import numpy as np
import scipy
from jax import numpy as jnp
from qpsolvers import solve_qp
from tqdm import tqdm

from ..preprocessing import UnVectorizer, Vectorizer
from ._base import BaseSM
from .utils import InitArrays, estimate_n_components_kmeans, nnsvd, sm_update_step, truncated_whitening


class NNCCA(BaseSM):
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

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def lag_step(self) -> int:
        return self._lag_step

    @property
    def lag_slices(self) -> list[slice]:
        if self.n_pixels is None:
            raise AttributeError("n_pixels is not set yet, must pre-initialize first")
        return [slice(self.n_pixels * i, (i + 1) * self.n_pixels) for i in range(self.lag * 2)]

    @property
    def movie(self) -> np.ndarray:
        if self._movie is None:
            raise AttributeError("not pre-initialized yet or pre-initialized without a movie")
        return self._movie

    @property
    def X_init(self) -> np.ndarray:
        if self._X_init is None:
            raise AttributeError(
                "not pre-initialized yet"
            )
        return self._X_init

    @property
    def unvec(self) -> UnVectorizer:
        if self._unvec is None:
            raise AttributeError(
                "not pre-initialized yet or pre-initialized without a movie"
            )
        return self._unvec

    @property
    def n_pixels(self) -> int:
        if self._n_pixels is None:
            raise AttributeError(
                "not pre-initialized yet"
            )
        return self._n_pixels

    @property
    def cov_init(self) -> np.ndarray:
        if self._cov_init is None:
            raise AttributeError("not pre-initialized yet")

        return self._cov_init

    @property
    def pre_init_arrays(self) -> InitArrays:
        if self._pre_init_arrays is None:
            raise AttributeError("not pre-initialized yet")
        return self._pre_init_arrays

    @property
    def init_arrays(self) -> InitArrays:
        if self._init_arrays is None:
            raise AttributeError("not pre-initialized yet")
        return self._init_arrays

    def pre_initialize(self, data: np.ndarray, max_k: int, k: int = None, k_add_trunc_w: int = 0, method: str = "nnsvd"):
        """
        Run nnSVD to pre-initialize Y before initializing with gradient descent.

        Parameters
        ----------
        data: data to initialize with, shape of [n_timepoints, rows, cols] or [n_pixels, n_timepoints]

        method: only "nnsvd" for now

        """
        if data.ndim == 3:
            self._movie = data
            self._X_init = Vectorizer().transform(self.movie)
            self._unvec = UnVectorizer(self.movie.shape[1:])
        elif data.ndim == 2:
            self._X_init = data
        else:
            raise ValueError("`data` ndim must be 2 or 3")

        if self.X_init.shape[0] * self.lag > self.X_init.shape[1]:
            raise ValueError(
                f"(n_pixels * lag) > timepoints. Require more timepoints to initialize\n"
                f"X_init shape: {self.X_init.shape}"
            )

        self._n_pixels = self._X_init.shape[0]

        if method != "nnsvd":
            raise ValueError("only nnsvd supported for now")

        Hs = list()

        print("creating H")
        for i in range(self.n_pixels):
            Hs.append(
                scipy.linalg.hankel(
                    self.X_init[i, : self.lag * self.lag_step * 2], self.X_init[i, (self.lag * self.lag_step * 2) - 1 :]
                )[::self.lag_step]
            )

        H = np.zeros((self.lag * 2 * self.n_pixels, self.X_init.shape[1] - (self.lag * self.lag_step * 2) + 1))

        for i, H_i in enumerate(Hs):
            H[i::self.n_pixels] = H_i

        past = H[: self.lag * self.n_pixels].copy()
        future = H[self.lag * self.n_pixels :].copy()

        mu0 = H[: self.lag * self.n_pixels].mean(axis=1)
        mu1 = H[self.lag * self.n_pixels :].mean(axis=1)

        H[: self.lag * self.n_pixels] -= mu0[:, None]
        H[self.lag * self.n_pixels :] -= mu1[:, None]

        past_mc = H[: self.lag * self.n_pixels].copy()
        future_mc = H[self.lag * self.n_pixels :].copy()


        if k is None:
            print("estimating k")
            k = estimate_n_components_kmeans(
                H_nw=np.vstack([past, future]),
                max_k=max_k,
            )

        print("Whitening H")

        # past_w = truncated_whitening(past, past_mc, k=k + k_add_trunc_w)
        # future_w = truncated_whitening(future, future_mc, k=k + k_add_trunc_w)
        #
        # Hw = np.vstack([past_w, future_w])

        Zp = scipy.linalg.pinv(scipy.linalg.sqrtm(past_mc @ past_mc.T / past_mc.shape[1]))
        Zf = scipy.linalg.pinv(scipy.linalg.sqrtm(future_mc @ future_mc.T / future_mc.shape[1]))

        Hw = np.vstack(
            [
                Zp @ past_mc,
                Zf @ future_mc,
            ]
        )

        print("computing covariance")
        self._cov_init = (Hw.T @ Hw) / Hw.shape[1]
        # cov = Hw.T @ Hw
        #
        # self._cov_init = cov / np.linalg.norm(cov, ord="fro")


        print("performing nnSVD")
        _, Y = nnsvd(A=self.cov_init, k=k)

        self._pre_init_arrays = InitArrays(
            Hw=Hw,
            H=np.vstack([past_mc, future_mc]),
            P=past,
            F=future,
            # Pw=past_w,
            # Fw=future_w,
            Zp=Zp,
            Zf=Zf,
            mu_p=mu0,
            mu_f=mu1,
            Y=Y,
            k=k,
            lag=self.lag,
            lag_step=self.lag_step,
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
            Y, err = sm_update_step(jnp.array(Y), jnp.array(self.cov_init), jnp.array(eta, dtype=jnp.float32))

            error_log.append(float(err))

            # below inertia threshold, error is decreasing and below error threshold
            if iteration > 2:
                if (
                    (error_log[-2] - error_log[-1]) < inertia
                    and (error_log[-1] < error_log[-2])
                    and (error_log[-1] < error_threshold)
                ):
                    break

        H_nw = np.vstack([self._pre_init_arrays.P, self._pre_init_arrays.F])

        Y = np.array(Y)

        W = Y @ self._pre_init_arrays.Hw.T  # / sum_y_squared
        W_nw = Y @ H_nw.T  # / sum_y_squared
        M = Y @ Y.T  # / sum_y_squared

        self._init_arrays = InitArrays(
            Hw=self._pre_init_arrays.Hw,
            P=self._pre_init_arrays.P,
            F=self._pre_init_arrays.F,
            mu_p=self._pre_init_arrays.mu_p,
            mu_f=self._pre_init_arrays.mu_f,
            Y=Y,
            k=self._pre_init_arrays.k,
            lag=self.lag,
            lag_step=self.lag_step,
            W=W,
            W_nw=W_nw,
            M=M,
        )

        return error_log, Ys, Ws

    # def fit_timepoint(self, x_t: np.ndarray) -> np.ndarray:
    #     """
    #
    #     Parameters
    #     ----------
    #     x_t: vector of shape [n_pixels], non-whitened, non-centered
    #
    #     Returns
    #     -------
    #     y_t: vector of shape [k, ]
    #
    #     """
    #
    #     if x_t.ndim > 1:
    #         x_t = x_t.ravel()
    #
    #     # whiten
    #     x_t = self.pre_init_arrays.Z @ x_t
    #
    #     x_t = x_t.astype(np.float64)
    #
    #     Wt = self.init_arrays.W.astype(np.float64)
    #     Mt = self.init_arrays.M.astype(np.float64)
    #     k = self.init_arrays.k
    #
    #     return solve_qp((Mt + Mt.T) / 2, -Wt @ x_t, lb=np.zeros(k), solver="cvxopt")
