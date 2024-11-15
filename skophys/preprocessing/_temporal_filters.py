import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin
from ._utils import Vectorizer, UnVectorizer



@numba.guvectorize([(numba.float32[:], numba.int32, numba.int32, numba.float32[:])], "(n), (), () -> (n)")
def _rolling_percentile(trace: np.ndarray, window_size: np.int32, quantile: np.int32, out: np.ndarray):
    for i in range(0, trace.size - 1):
        out[i] = np.percentile(trace[i:i + window_size], quantile)


@numba.njit(nopython=True, parallel=True)
def _rolling_percentile_movie(Y: np.ndarray, window_size: int, quantile: int, output: np.ndarray):
    window_size = np.int32(window_size)
    quantile = np.int32(quantile)

    for i in numba.prange(Y.shape[0]):
        _rolling_percentile(Y[i], window_size, quantile, output[i])


def rolling_percentile(
        Y: np.ndarray,
        window_size: int = 10,
        quantile: int = 5,
) -> np.ndarray:
    """
    Apply a rolling percentile filter on a movie
    Parameters
    ----------
    Y: np.ndarray, shape [n_pixels, time]
        vectorized movie

    Returns
    -------
    np.ndarray
        percentile filtered movie, shape [n_pixels, time]

    """

    filtered_movie = np.zeros(Y.shape, dtype=np.float32)

    _rolling_percentile_movie(
        Y.astype(np.float32),
        window_size,
        quantile,
        filtered_movie
    )

    return filtered_movie


class PercentileFilter(TransformerMixin, BaseEstimator):
    """
    Rolling percentile filter across time

    Parameters
    ----------
    window_size: int, default 10
        window size

    quantile: int, default 5
        quantile
    """

    def __init__(self, window_size: int = 10, quantile: int = 5):
        self.window_size = window_size
        self.quantile = quantile

    def fit(self, movie, y=None):
        """
        Does nothing, exists for API conformity
        """

        return self

    def transform(self, movie):
        """
        Apply the percentile filter

        Parameters
        ----------
        movie : array-like, shape: [n_pixels, time] or [time, rows, cols]
            input movie

        Returns
        -------
        np.ndarray
            filtered movie, shape [n_pixels, time]
        """

        Y = Vectorizer().transform(movie)
        filtered = rolling_percentile(Y, self.window_size, self.quantile)

        return filtered

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}
