from typing import Literal
from copy import deepcopy

import numpy as np


class ARProcess:
    def __init__(
            self,
            n_timepoints: int,
            n_components: int,
            dimensions: int,
            firing_probability: float = 0.005,
            n_pixels_per_component: int = 1,
            spikes: np.ndarray = None,
            obs_noise_sigma: float = 0.0,
            ar_noise_sigma: float = 0.0,
            n_background_pixels: int = 0,
            interpolation_factor: int = 1,
            random_seed: int = 0,
            rise_constant: float | tuple[float] = 5.0,
            decay_constant: float | tuple[float] = 0.7,
            # decay_matrix: np.ndarray[float] = None,
            # decay_matrix_jitter_sigma: float = 0.0,
            rise_time: int = 10,
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
        dimensions
        firing_probability
        spikes
        obs_noise_sigma
        n_background_pixels
        interpolation_factor
        random_seed
        rise_constant:
        decay_constant: scaler decay constant
        decay_matrix: decay matrix, optional
            provide decay matrix or scaler decay constant
        decay_matrix_jitter_sigma: float, default 0.0

        rise_time
        """
        self._params = {
            "n_timepoints": n_timepoints,
            "n_components": n_components,
            "dimensions": dimensions,
            "firing_probability": firing_probability,
            "n_pixels_per_component": n_pixels_per_component,
            "spikes": spikes,
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
        # print(clean)

        if isinstance(decay_constant, float):
            a_options = [decay_constant]
        elif isinstance(decay_constant, tuple):
            a_options = decay_constant
        else:
            raise TypeError("`decay_constant` must be a float or a tuple")

        a_decay = decay_constant[0]
        a_rise = 5

        b = 0

        rise_time = 10

        n_rise = 0

        spikes = list()

        for i in range(n_components):
            spikes.append((np.random.rand(n_timepoints) < firing_probability).astype(bool))

        for c_ix in range(n_components):
            for i in range(2, n_timepoints):
                if spikes[c_ix][i]:
                    a = a_rise
                    a_decay = np.random.choice(a_options)
                    n_rise = rise_time
                elif n_rise > 0:
                    if clean[c_ix, i - 1] > 1.5:
                        a = a_decay
                        n_rise = 0
                    else:
                        a = a_rise
                        n_rise -= 1
                else:
                    a = a_decay
                clean[c_ix, i] = (a * clean[c_ix, i - 1]) + (b * clean[c_ix, i - 2]) + np.abs(
                    np.random.normal(scale=ar_noise_sigma, size=1))

            x_ = np.arange(0, n_timepoints * interpolation_factor)
            xp_ = np.linspace(0, n_timepoints * interpolation_factor, n_timepoints)
            clean[c_ix] = np.interp(x_, xp_, fp=clean[c_ix, :n_timepoints])

        n_timepoints *= interpolation_factor

        series = list()
        labels = list()

        for i in range(n_components):
            # determine number of pixels for this component
            if isinstance(n_pixels_per_component, int):
                _n_pixels_iter = n_pixels_per_component
            elif isinstance(n_pixels_per_component, tuple):
                _n_pixels_iter = np.random.randint(low=n_pixels_per_component[0], high=n_pixels_per_component[1])

            for j in range(_n_pixels_iter):
                series.append(clean[i] + np.random.normal(scale=obs_noise_sigma, size=n_timepoints))
                labels.append(i + 1)

        # series = np.vstack(series)

        bg_pixels = [np.random.normal(scale=obs_noise_sigma, size=n_timepoints) for i in range(n_background_pixels)]
        series = np.vstack([*series, *bg_pixels])
        labels = np.array([*labels, *[0 for i in range(len(bg_pixels))]])


    @property
    def params(self) -> dict:
        return deepcopy(self._params)

    @property
    def spikes(self) -> np.ndarray:
        """spikes ground truth"""
        pass

    @property
    def trace(self) -> np.ndarray:
        """trace ground truth without observation noise"""
        pass

    @property
    def components(self) -> np.ndarray:
        """component labels, -1 indicates a backgound component, shape is [n_pixels]"""
        pass

    def to_hdf5(self):
        pass

    @classmethod
    def from_hdf5(cls):
        pass


class ARProcessMovie(ARProcess):
    def __init__(
            self,
            component_shape: Literal["rect", "gaussian"] = "gaussian",
            movie_dims: tuple[int, int] = (30, 30),
            component_size: tuple[int, int] = (10, 10),
            component_locs: np.ndarray[int] | Literal["random"] = "random",
            component_locs_random_seed: int = 0,
            **kwargs,
    ):
        super().__init__(**kwargs)

    @classmethod
    def from_1d_model(cls):
        pass

    def to_hdf5(self):
        pass

    @classmethod
    def from_hdf(cls):
        pass
