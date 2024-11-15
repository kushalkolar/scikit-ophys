import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def vectorize(movie, order="C"):
    """
    [time, rows, cols] -> [n_pixels, time]
    """
    return movie.transpose(1, 2, 0).reshape(np.prod(movie.shape[1:]), movie.shape[0], order=order)


def unvectorize(Y, shape, order="C"):
    """
    [n_pixels, time] -> [time, rows, cols]
    """
    return Y.reshape(*shape, Y.shape[1], order=order).transpose(-1, 0, 1)


class Vectorizer(TransformerMixin, BaseEstimator):
    """
    Vectorize movies

    Parameters
    ----------
    order: str, array order
        "C" or "F"

    """

    def __init__(self, order: str = "C"):
        self.order = order

    def fit(self, movie, y=None):
        """
        Does nothing, exists for API conformity
        """

        return self

    def transform(self, movie: np.ndarray):
        """
        Vectorize the movie. Does nothing if the movie is already vectorized.

        Parameters
        ----------
        movie : array-like, shape [time, rows, cols]
            input movie

        Returns
        -------
        np.ndarray
            vectorized movie, shape [n_pixels, time]

        """

        if movie.ndim == 2:
            return movie

        return vectorize(movie, order=self.order)

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}


class UnVectorizer(TransformerMixin, BaseEstimator):
    """
    Unvectorize movies

    Parameters
    ----------
    shape: tuple, [n_rows, n_cols]
        shape of a 2D frame of the movie

    order: str, array order
        "C" or "F"

    """

    def __init__(self, shape: tuple, order: str = "C"):
        if len(shape) != 2:
            raise ValueError
        self.shape = shape

        self.order = order

    def fit(self, movie, y=None):
        """
        Does nothing, exists for API conformity
        """

        return self

    def transform(self, Y: np.ndarray):
        """
        Unvectorize the movie. Does nothing if the movie is already unvectorized.

        Parameters
        ----------
        Y: array-like, shape [n_pixels, time]
            input movie

        Returns
        -------
        np.ndarray
            unvectorized movie, shape [time, rows, cols]

        """

        if Y.ndim == 3:
            return Y

        return unvectorize(Y, shape=self.shape, order=self.order)

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}
