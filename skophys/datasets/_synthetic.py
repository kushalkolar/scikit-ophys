from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np

def eigen_decomposition(M):
    # output eigenvalues and eigenvectors sorted by eigenvalues
    
    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    print (f"eigenvectors.shape = {eigenvectors.shape}")

    return eigenvalues, eigenvectors

def covariance_matrix(X):
    # we assume X has a shape (N,T)
    # N = number of features
    # T = number of samples
    
    T = X.shape[1]
    COV = X @ X.T / float(T)
    
    return COV

def whitened_matrix(X, X0, k):
    # we assume X has a shape (N,T)
    # N = number of features
    # T = number of samples

    # COV shape = (N, N)
    COV = covariance_matrix(X0)
    S, U = eigen_decomposition(COV)

    S_eco = S[:k]
    U_eco = U[:, :k]

    # Epsilon is added to the eigenvalues to prevent division by zero
    epsilon = 1e-5
    inv_sqrt_S = np.diag(1.0 / np.sqrt(S_eco + epsilon))

    # whitening matrix W shape (N, N)
    W = U_eco @ inv_sqrt_S @ U_eco.T

    # whitened data shape (N, T)
    Xw = W @ X

    return Xw

class ARProcess:
    def __init__(
        self,
        n_timepoints: int,
        n_components: int,
        firing_probability: float = 0.005,
        n_pixels_per_component: int = 1,
        obs_noise_sigma: float = 0.1,
        ar_noise_sigma: float = 0.1,
        ar_bias: float = 0.0,
        n_background_pixels: int = 0,
        interpolation_factor: int = 1,
        random_seed: int = 0,
        rise_constant: float = 5.0,
        decay_constant: float | tuple[float] = 0.7,
        # decay_matrix: np.ndarray[float] = None,
        # decay_matrix_jitter_sigma: float = 0.0,
        rise_time: int = 10,
        spikes: np.ndarray = None,
        traces: np.ndarray = None,
        traces_noisy: np.ndarray = None,
        labels: np.ndarray = None,
    ):
        """
        Produces a synthetic dataset using an auto-regressive model.

        For each component:

        x_{t + 1} = a * x_t + ε

        where:
            a: decay_constant | rise_constant
            ε: noise sampled from a normal distribution with standard deviation: ``ar_noise_sigma``

            x and a are scalers

        Parameters
        ----------
        n_timepoints
        n_components
        firing_probability
        obs_noise_sigma
        n_background_pixels
        interpolation_factor
        random_seed
        rise_constant:
        decay_constant: scaler decay constant
        rise_time
        spikes
        """
        self._params = {
            "n_timepoints": n_timepoints,
            "n_components": n_components,
            "firing_probability": firing_probability,
            "n_pixels_per_component": n_pixels_per_component,
            "noise_sigma": obs_noise_sigma,
            "n_background_pixels": n_background_pixels,
            "interpolation_factor": interpolation_factor,
            "random_seed": random_seed,
            "rise_constant": rise_constant,
            "decay_constant": decay_constant,
            # "decay_matrix": decay_matrix,
            # "decay_matrix_jitter_sigma": decay_matrix_jitter_sigma,
            "rise_time": rise_time,
        }

        clean = np.zeros((n_components, n_timepoints * interpolation_factor)) + 0.01

        if isinstance(decay_constant, float):
            decay_constant = [decay_constant]
        elif isinstance(decay_constant, (tuple, list, np.ndarray)):
            decay_constant = decay_constant
        else:
            raise TypeError("`decay_constant` must be a float or one of: tuple, list, array")

        a_options = decay_constant
        a_decay = a_options[0]
        a_rise = rise_constant

        b = 0

        n_rise = 0

        if spikes is None:
            spikes = np.zeros((n_components, (n_timepoints) * interpolation_factor), dtype=bool)

            np.random.seed(random_seed)

            for k_i in range(n_components):
                spikes[k_i] = (np.random.rand(n_timepoints) < firing_probability)

            self._spikes = spikes
        else:
            self._spikes = spikes

        if traces is None:
            for k_i in range(n_components):
                ar_noise_sigma_k_i = np.random.uniform(low=0.0, high=ar_noise_sigma)
                for t_i in range(2, n_timepoints):
                    #gamma = np.array([1.5, -.55])
                    #clean[k_i, t_i] = (gamma[0] * clean[k_i, t_i - 1]) + (gamma[1] * clean[k_i, t_i - 2])
                    if self.spikes[k_i, t_i]:
                        a = a_rise
                        a_decay = np.random.choice(a_options)
                        n_rise = rise_time
                    elif n_rise > 0:
                        if clean[k_i, t_i - 1] > 1.5:
                            a = a_decay
                            n_rise = 0
                        else:
                            a = a_rise
                            n_rise -= 1
                    else:
                        a = a_decay
                    clean[k_i, t_i] = (a * clean[k_i, t_i - 1]) + (b * clean[k_i, t_i - 2]) + np.abs(
                        np.random.normal(scale=ar_noise_sigma_k_i, size=1))

                x_ = np.arange(0, (n_timepoints) * interpolation_factor)
                xp_ = np.linspace(0, (n_timepoints) * interpolation_factor, (n_timepoints))
                clean[k_i] = np.interp(x_, xp_, fp=clean[k_i, :n_timepoints])

            #clean = whitened_matrix(clean, clean, n_components)
            for k_i in range(n_components):
                bias = np.random.uniform(low=0.0, high=ar_bias)
                for t_i in range(2, n_timepoints):
                    clean[k_i, t_i] += bias
            self._traces = clean.copy()

            n_timepoints *= interpolation_factor

            trace = list()
            labels = list()

            for k_i in range(n_components):
                # determine number of pixels for this component
                if isinstance(n_pixels_per_component, int):
                    _n_pixels_iter = n_pixels_per_component
                elif isinstance(n_pixels_per_component, tuple):
                    _n_pixels_iter = np.random.randint(
                        low=n_pixels_per_component[0], high=n_pixels_per_component[1]
                    )

                for j in range(_n_pixels_iter):
                    trace.append(
                        clean[k_i]
                        + np.random.normal(scale=obs_noise_sigma, size=n_timepoints)
                    )
                    labels.append(k_i + 1)

            bg_pixels = [
                np.random.normal(scale=obs_noise_sigma, size=n_timepoints)
                for i in range(n_background_pixels)
            ]

            self._traces_obs = np.vstack([*trace, *bg_pixels])
            self._labels = np.array([*labels, *[-1 for i in range(len(bg_pixels))]])
        else:
            if (traces_noisy is None) or (labels is None):
                raise ValueError("must provide `traces_noisey` and `labels` if providing `traces`")

            self._traces = traces
            self._traces_obs = traces_noisy
            self._labels = labels

    @property
    def params(self) -> dict:
        return deepcopy(self._params)

    @property
    def spikes(self) -> np.ndarray:
        """spikes ground truth"""
        return self._spikes

    @property
    def traces(self) -> np.ndarray:
        """trace ground truth with observation noise"""
        return self._traces

    @property
    def traces_obs(self) -> np.ndarray:
        """traces with observation noise, shape is [n_components * n_pixels_per_component, n_timepoints * interpolation_factor]"""
        return self._traces_obs

    @property
    def labels(self) -> np.ndarray:
        """component labels, -1 indicates a backgound component, shape is [n_pixels]"""
        return self._labels

    def to_npz(self, path: str | Path):
        np.savez(
            path,
            spikes=self.spikes,
            traces=self.traces,
            traces_noisy=self.traces_obs,
            labels=self.labels,
            params=self.params,
        )

    @classmethod
    def from_npz(cls, path: str | Path):
        data = dict(np.load(path))

        for k in data.keys():
            if data[k].size == 1:
                data[k] = data[k].item()

        return cls(**data)

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


class ARProcessMovie(ARProcess):
    def __init__(
            self,
            component_shape: Literal["rect", "gaussian"] = "gaussian",
            movie_dims: tuple[int, int] = (32, 32),
            component_size: int = 2,
            component_locs: np.ndarray[int] | Literal["random"] = "random",
            component_locs_random_seed: int = 0,
            **kwargs,
    ):
        super().__init__(**kwargs)

        r = -2*component_size, 2*component_size
        r = np.linspace(*r, num=4*component_size+2)
        print (f"r.shape = {r.shape}")
        x, y = np.meshgrid(r, r)
        z = gaus2d(x, y, sx=component_size, sy=component_size)
        z /= z.max()
        print (f"z.shape = {z.shape}")

        #r = -20, 20
        #r = np.linspace(*r)
        #x, y = np.meshgrid(r, r)
        #z = gaus2d(x, y)[20:-20, 20:-20]
        #z /= z.max()
        #print (z.shape)

        if isinstance(component_locs, str):
            if component_locs == "random":
                np.random.seed(component_locs_random_seed)
                #component_locs = (np.random.rand(kwargs["n_components"], 2) * movie_dims[0] - z.shape[0]).clip(0).astype(int)
                component_locs = (np.random.rand(100, 2) * movie_dims[0]).clip(0).astype(int)
            else:
                raise ValueError("invalid value for `component_locs`")

        k = self.params["n_components"]

        spatial_footprints = np.zeros((k, *movie_dims))
        print (f"spatial_footprints.shape = {spatial_footprints.shape}")

        #for i in range(k):
        #    cols = slice(component_locs[i, 0], component_locs[i, 0] + 10)
        #    rows = slice(component_locs[i, 1], component_locs[i, 1] + 10)
        #    spatial_footprints[i, rows, cols] = z

        added_source = 0
        for i in range(100):
            if added_source==k: break
            cols = slice(component_locs[i, 0] - int(z.shape[0]/2), component_locs[i, 0] + int(z.shape[1]/2))
            rows = slice(component_locs[i, 1] - int(z.shape[0]/2), component_locs[i, 1] + int(z.shape[1]/2))
            if z.shape == spatial_footprints[added_source, rows, cols].shape:
                spatial_footprints[added_source, rows, cols] = z
                added_source += 1
        print (f"added_source = {added_source}")

        serial_spatial_footprints = spatial_footprints.reshape(k, np.prod(movie_dims))

        X = self.traces

        self.kernel = z.copy()
        self.source_footprints = spatial_footprints.copy()

        self._movie = (serial_spatial_footprints.T @ X).reshape(*movie_dims, X.shape[1]).transpose(-1, 0, 1).copy()

        noise_sigma = kwargs["obs_noise_sigma"]
        noise = np.random.normal(scale=noise_sigma, size=np.prod(self._movie.shape)).reshape(self._movie.shape)
        self._movie += noise

    @classmethod
    def from_1d_model(
            cls,
            model: ARProcess,
            component_shape: Literal["rect", "gaussian"] = "gaussian",
            movie_dims: tuple[int, int] = (30, 30),
            component_size: tuple[int, int] = (10, 10),
            component_locs: np.ndarray[int] | Literal["random"] = "random",
            component_locs_random_seed: int = 0,
    ):
        return cls(
            component_shape=component_shape,
            movie_dims=movie_dims,
            component_size=component_size,
            component_locs=component_locs,
            component_locs_random_seed=component_locs_random_seed,
            **model.params
        )


    @property
    #def source_footprints(self) -> np.ndarray:
    #    """source_footprints, [k, rows, cols]"""
    #    return self._source_footprints

    #def traces(self) -> np.ndarray:
    #    """traces, [rows, cols]"""
    #    return self._traces

    #def kernel(self) -> np.ndarray:
    #    """kernel, [rows, cols]"""
    #    return self._kernel

    def movie(self) -> np.ndarray:
        """movie, [t, rows, cols]"""
        return self._movie

    def to_hdf5(self):
        pass

    @classmethod
    def from_hdf5(cls):
        pass
