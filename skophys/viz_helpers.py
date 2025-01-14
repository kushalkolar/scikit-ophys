import numpy as np


def spatial_to_rgba(spatial_filters: np.ndarray, percentile: float = 90.0):
    """
    Get an RGBA array that represents the given spatial filters

    Parameters
    ----------
    spatial_filters: np.ndarray
        shape: [n_components, n_rows, n_cols]

    percentile: float, default 90.0
        only pixels above this percentile are considered

    Returns
    -------
    [n_components, n_rows, n_cols, 4], last dim is rgba

    """
    
    out = np.zeros((*spatial_filters.shape, 4))

    for i, a in enumerate(spatial_filters):
        rgba_component = np.zeros((*a.shape, 4))
        # threshold using percentile
        rows, cols = np.where(a > np.percentile(a, percentile))

        # set a random color
        rgba_component[rows, cols, :-1] = np.random.rand(3)

        # set alpha value as the filter value
        rgba_component[rows, cols, -1] = a[rows, cols]

        out[i] = rgba_component

    return out
