"""Feature transformers for MNIST (flattened 28x28 images)."""

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.base import BaseEstimator, TransformerMixin

IMAGE_SHAPE = (28, 28)
N_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


class GaussianBlur(BaseEstimator, TransformerMixin):
    """Blur each flattened 28x28 image with an isotropic Gaussian filter.

    The transform is stateless and linear: x' = G_sigma @ x.
    sigma = 0 is the identity, so the raw-pixel baseline goes through the
    same pipeline as every blurred configuration.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel, in pixels.
    """

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != N_PIXELS:
            raise ValueError(f"Expected shape (n_samples, {N_PIXELS}), got {X.shape}")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.sigma == 0:
            return X
        imgs = X.reshape(-1, *IMAGE_SHAPE)
        # sigma=0 on axis 0 -> each image is blurred independently
        blurred = gaussian_filter(
            imgs, sigma=(0, self.sigma, self.sigma), mode="constant", cval=0.0
        )
        return blurred.reshape(len(X), -1)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags
