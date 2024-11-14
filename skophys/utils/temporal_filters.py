import numpy as np
import numba


@numba.guvectorize([(numba.float32[:], numba.int32, numba.int32, numba.float32[:])], "(n), (), () -> (n)")
def _rolling_percentile(trace: np.ndarray, window_size: np.int32, quantile: np.int32, out: np.ndarray):
    for i in range(0, trace.size - 1):
        out[i] = np.percentile(trace[i:i + window_size], quantile)


@numba.njit(nopython=True, parallel=True)
def _rolling_percentile_movie(raw_movie: np.ndarray, window_size: int, quantile: int, output: np.ndarray):
    t, row, col = raw_movie.shape

    window_size = np.int32(window_size)
    quantile = np.int32(quantile)

    for i in numba.prange(row):
        for j in range(col):
            _rolling_percentile(raw_movie[:, i, j], window_size, quantile, output[:, i, j])


def rolling_percentile(
        movie: np.ndarray,
        window_size: int = 10,
        quantile: int = 5,
) -> np.ndarray:
    """
    Apply a rolling percentile filter on a movie
    Parameters
    ----------
    movie

    Returns
    -------

    """
    filtered_movie = np.zeros(movie.shape, dtype=np.float32)

    _rolling_percentile_movie(
        movie.astype(np.float32),
        window_size,
        quantile,
        filtered_movie
    )

    return filtered_movie
