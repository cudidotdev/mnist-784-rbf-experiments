from pathlib import Path

import numpy as np
import polars as pl
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

MNIST_DATA_PATH = MNIST_DIR / "mnist.parquet"


def download_mnist(force: bool = False) -> pl.DataFrame:
    """
    Download MNIST from OpenML and persist it locally as Parquet.

    Parameters
    ----------
    force : bool
        If True, download the dataset again, even if a local
        Parquet copy already exists.

    Returns
    -------
    pl.DataFrame
        MNIST dataset containing 784 pixel columns and a label column.
    """

    # Create data directories
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse local dataset unless forced to download again
    if MNIST_DATA_PATH.exists() and not force:
        print("MNIST already exists locally.")
        return pl.read_parquet(MNIST_DATA_PATH)

    print("Downloading MNIST from OpenML...")

    mnist = fetch_openml(
        data_id=MNIST_OPENML_ID,
        as_frame=False,
        parser="liac-arff",
        data_home=str(OPENML_CACHE_DIR),
        cache=True,
    )

    X = np.asarray(mnist.data)
    y = np.asarray(mnist.target)

    X = X.astype(np.uint8)
    y = y.astype(np.uint8)

    # Construct Polars DataFrame
    pixel_columns = [f"pixel_{i}" for i in range(X.shape[1])]

    df = pl.DataFrame(
        X,
        schema=pixel_columns,
        orient="row",
    ).with_columns(pl.Series("label", y))

    # Persist dataset
    df.write_parquet(
        MNIST_DATA_PATH,
        compression="zstd",
    )

    print("MNIST downloaded and saved successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset path: {MNIST_DATA_PATH}")

    return df


def load_mnist(download_if_missing: bool = True) -> pl.DataFrame:
    """
    Load MNIST from local Parquet storage.

    Downloads the dataset from OpenML if it does not exist locally
    and download_if_missing is True.

    Parameters
    ----------
    download_if_missing : bool
        Whether to download the dataset when it is not available locally.

    Returns
    -------
    pl.DataFrame
        MNIST dataset.
    """

    if MNIST_DATA_PATH.exists():
        print("Loading MNIST from local storage.")
        return pl.read_parquet(MNIST_DATA_PATH)

    if not download_if_missing:
        raise FileNotFoundError(f"MNIST dataset not found at {MNIST_DATA_PATH}")

    return download_mnist()


N_TRAIN = 60_000
N_TOTAL = 70_000
PIXEL_COLS = [f"pixel_{i}" for i in range(784)]

# Official MNIST class counts, indexed by digit 0-9
EXPECTED_TRAIN_COUNTS = np.array(
    [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
)
EXPECTED_TEST_COUNTS = np.array([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009])


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
    df = load_mnist()
    X = df.select(PIXEL_COLS).to_numpy()
    y = df["label"].to_numpy()

    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    _check_split_integrity(y_train, y_test)

    if scale:
        X = X.astype(dtype) / 255.0
    return X[:N_TRAIN], X[N_TRAIN:], y_train, y_test
