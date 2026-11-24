from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

MNIST_OPENML_ID = 554

# Root of the project (works with an editable installation)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# MNIST-specific paths
MNIST_DIR = RAW_DATA_DIR / "mnist"
OPENML_CACHE_DIR = MNIST_DIR / "openml_cache"

MNIST_DATA_PATH = MNIST_DIR / "mnist.npz"

N_TRAIN = 60_000
N_TOTAL = 70_000
N_FEATURES = 784

# Official MNIST class counts, indexed by digit 0-9
EXPECTED_TRAIN_COUNTS = np.array(
    [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
)
EXPECTED_TEST_COUNTS = np.array([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009])


def _read_npz() -> tuple[np.ndarray, np.ndarray]:
    """Read the raw uint8 arrays from the local .npz file."""
    with np.load(MNIST_DATA_PATH) as data:
        return data["X"], data["y"]


def _scale(X: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Convert uint8 pixels in [0, 255] to floats in [0, 1]."""
    return X.astype(dtype) / dtype(255)


def download_mnist(force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Download MNIST from OpenML and persist it locally as a compressed .npz.

    The file stores the raw, unscaled data:
      - X: uint8, shape (70000, 784), values in [0, 255]
      - y: uint8, shape (70000,), values in [0, 9]

    Parameters
    ----------
    force : bool
        If True, download the dataset again, even if a local
        copy already exists.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Raw (X, y) as uint8 arrays.
    """

    # Create data directories
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse local dataset unless forced to download again
    if MNIST_DATA_PATH.exists() and not force:
        print("MNIST already exists locally.")
        return _read_npz()

    print("Downloading MNIST from OpenML...")

    mnist = fetch_openml(
        data_id=MNIST_OPENML_ID,
        as_frame=False,
        parser="liac-arff",
        data_home=str(OPENML_CACHE_DIR),
        cache=True,
    )

    X = np.asarray(mnist.data).astype(np.uint8)
    y = np.asarray(mnist.target).astype(np.uint8)

    # Persist dataset
    np.savez_compressed(MNIST_DATA_PATH, X=X, y=y)

    print("MNIST downloaded and saved successfully.")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    print(f"Dataset path: {MNIST_DATA_PATH}")

    return X, y


def load_mnist(
    scale: bool = True,
    dtype=np.float32,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the full MNIST dataset (all 70k images) from local storage.

    Downloads the dataset from OpenML if it does not exist locally
    and download_if_missing is True.

    Parameters
    ----------
    scale : bool
        If True, return X as floats in [0, 1]. If False, return the raw
        uint8 pixels in [0, 255].
    dtype : numpy dtype
        Float dtype used when scale is True. Ignored otherwise.
    download_if_missing : bool
        Whether to download the dataset when it is not available locally.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X with shape (70000, 784) and y with shape (70000,), uint8 labels.
        The first N_TRAIN rows are the official training split.
    """

    if MNIST_DATA_PATH.exists():
        print("Loading MNIST from local storage.")
        X, y = _read_npz()
    elif download_if_missing:
        X, y = download_mnist()
    else:
        raise FileNotFoundError(f"MNIST dataset not found at {MNIST_DATA_PATH}")

    if scale:
        X = _scale(X, dtype)
    return X, y


def _check_split_integrity(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Fail loudly if the row ordering does not match the official MNIST split."""
    assert len(y_train) == N_TRAIN and len(y_test) == N_TOTAL - N_TRAIN, (
        f"Unexpected split sizes: {len(y_train)} / {len(y_test)}"
    )
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)
    assert np.array_equal(train_counts, EXPECTED_TRAIN_COUNTS), (
        f"Train class counts differ from official MNIST: {train_counts}"
    )
    assert np.array_equal(test_counts, EXPECTED_TEST_COUNTS), (
        f"Test class counts differ from official MNIST: {test_counts}"
    )


def load_mnist_split(scale: bool = True, dtype=np.float32):
    """Return X_train, X_test, y_train, y_test using the official 60k/10k split."""
    # Load unscaled, split first, then scale, to avoid an extra full-size float copy
    X, y = load_mnist(scale=False)

    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    _check_split_integrity(y_train, y_test)

    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    if scale:
        X_train, X_test = _scale(X_train, dtype), _scale(X_test, dtype)
    return X_train, X_test, y_train, y_test
