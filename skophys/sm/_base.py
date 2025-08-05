import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from ..preprocessing import UnVectorizer
from .utils import InitArrays, get_permute_indices
from ..datasets import ARProcess


class BaseSM:
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
        raise NotImplementedError

    def initialize(
            self,
            max_iter: int = 2_000,
            eta: float = 0.1,
            error_threshold: float = 1e-3,
            keep_iteration_results: bool = False,
    ):
        raise NotImplementedError

    def permute_indices(self, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        permute indices of Y, W, M, to match ground truth from AR model

        Returns
        -------
        permutation_indices, cosine_similarity matrix

        """
        cos_sim = cosine_similarity(
            ground_truth, self.init_arrays.Y
        )

        ixs_permute = get_permute_indices(cos_sim)
        cos_sim = cos_sim[:, ixs_permute]

        self.init_arrays.Y = self.init_arrays.Y[ixs_permute]

        W = self.init_arrays.Y @ self._pre_init_arrays.Pw.T
        M = self.init_arrays.Y @ self.init_arrays.Y.T

        self.init_arrays.W = W
        self.init_arrays.M = M

        return ixs_permute, cos_sim