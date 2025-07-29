from dataclasses import dataclass

import numpy as np
import scipy
from tqdm import tqdm

from ..preprocessing import Vectorizer, UnVectorizer
from .utils import estimate_n_components_kmeans, nnsvd


@dataclass
class InitArrays:
    Hw: np.ndarray  # whitened vstack([P, F])

    P: np.ndarray   # past lag vectors, not whitened
    F: np.ndarray   # future lag vectors, not whitened

    mu_p: np.ndarray  # mean of the past, shape is [n_pixels]
    mu_f: np.ndarray  # mean of the future, shape is [n_pixels]

    Y: np.ndarray   # output, shape is [k, n_timepoints]

    W: np.ndarray   # W matrix
    W_nw: np.ndarray  # W matrix with non-whitened data projected onto Y
    M: np.ndarray   # M matrix

    k: int                 # number of components
    lag: int
    lag_step: int


class BaseSM:
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

    def pre_initialize(self, data: np.ndarray, max_k: int, k: int = None, method: str = "nnsvd"):
        raise NotImplementedError

    def initialize(
            self,
            max_iter: int = 2_000,
            eta: float = 0.1,
            error_threshold: float = 1e-3,
            keep_iteration_results: bool = False,
    ):
        raise NotImplementedError

    def fit_timepoint(self, x_t: np.ndarray):
        raise NotImplementedError


class NNSM(BaseSM):
    def __init__(
            self,
    ):
        self._movie: np.ndarray = None
        self._X_init: np.ndarray = None
        self._unvec: UnVectorizer = None

        self._n_pixels: int = None

        self._cov_init: np.ndarray = None

        self._pre_init_arrays: InitArrays = None
        self._init_arrays: InitArrays = None


    def pre_initialize(self, data: np.ndarray, max_k: int, k: int = None, method: str = "nnsvd"):
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

        # mean center
        self._X_init = self.X_init - self.X_init.mean(axis=1)[:, None]

        self._n_pixels = self._X_init.shape[0]

        if method != "nnsvd":
            raise ValueError("only nnsvd supported for now")

        print("computing covariance")
        cov = self.X_init.T @ self.X_init

        self._cov_init = cov / np.linalg.norm(cov, ord="fro")

        if k is None:
            print("estimating k")
            k = estimate_n_components_kmeans(
                H_nw=self.X_init,
                max_k=max_k,
            )

        print("performing nnSVD")
        _, Y = nnsvd(A=self.cov_init, k=k)

        self._pre_init_arrays = InitArrays(
            Hw=None,
            P=None,
            F=None,
            mu_p=None,
            mu_f=None,
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
            error_threshold: float = 1e-3,
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
            if keep_iteration_results:
                Ys.append(Y)
                Ws.append(Y @ self.cov_init.T)

            # if iteration > 500:
            #     eta = 1.0
            # if iteration > 1_000:
            #     eta = 0.5

            Y_old = Y
            Y = Y + eta * Y @ (self.cov_init - Y.T @ Y)

            # zeroed = Y.sum(axis=1) == 0.0
            # Y[zeroed] = (np.random.randn(zeroed.sum(), T) / np.sqrt(T)).clip(0)

            Y = Y.clip(0)

            er = np.divide(np.abs(Y - Y_old), np.abs(Y_old + 1e-2)).max() / eta
            error_log.append(er)
            if er < error_threshold:
                break

        W = Y @ self.X_init.T  # / sum_y_squared
        M = Y @ Y.T  # / sum_y_squared

        self._init_arrays = InitArrays(
            Hw=None,
            P=None,
            F=None,
            mu_p=None,
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

    def fit_timepoint(self, x_t: np.ndarray):
        """

        Parameters
        ----------
        x_t: vector of shape [n_pixels]

        Returns
        -------

        """
        pass



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

    def pre_initialize(self, data: np.ndarray, max_k: int, k: int = None, method: str = "nnsvd"):
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

        mu0 = H[: self.lag * self.n_pixels].mean(axis=1)
        mu1 = H[self.lag * self.n_pixels :].mean(axis=1)

        H[: self.lag * self.n_pixels] -= mu0[:, None]
        H[self.lag * self.n_pixels :] -= mu1[:, None]

        past = H[: self.lag * self.n_pixels]
        future = H[self.lag * self.n_pixels :]

        print("Whitening H")
        Hw = np.vstack(
            [
                scipy.linalg.pinv(scipy.linalg.sqrtm(past @ past.T / past.shape[1])) @ past,
                scipy.linalg.pinv(scipy.linalg.sqrtm(future @ future.T / future.shape[1])) @ future,
            ]
        )

        print("computing covariance")
        cov = Hw.T @ Hw

        self._cov_init = cov / np.linalg.norm(cov, ord="fro")

        if k is None:
            print("estimating k")
            k = estimate_n_components_kmeans(
                H_nw=np.vstack([past, future]),
                max_k=max_k,
            )

        print("performing nnSVD")
        _, Y = nnsvd(A=self.cov_init, k=k)

        self._pre_init_arrays = InitArrays(
            Hw=Hw,
            P=past,
            F=future,
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
            error_threshold: float = 1e-3,
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
            if keep_iteration_results:
                Ys.append(Y)
                Ws.append(Y @ self.cov_init.T)

            # if iteration > 500:
            #     eta = 1.0
            # if iteration > 1_000:
            #     eta = 0.5

            Y_old = Y
            Y = Y + eta * Y @ (self.cov_init - Y.T @ Y)

            # zeroed = Y.sum(axis=1) == 0.0
            # Y[zeroed] = (np.random.randn(zeroed.sum(), T) / np.sqrt(T)).clip(0)

            Y = Y.clip(0)

            er = np.divide(np.abs(Y - Y_old), np.abs(Y_old + 1e-2)).max() / eta
            error_log.append(er)
            if er < error_threshold:
                break

        H_nw = np.vstack([self._pre_init_arrays.P, self._pre_init_arrays.F])

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

    def fit_timepoint(self, x_t: np.ndarray):
        """

        Parameters
        ----------
        x_t: vector of shape [n_pixels]

        Returns
        -------

        """
        pass
