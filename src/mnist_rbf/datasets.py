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
    pixel_columns = [
        f"pixel_{i}" for i in range(X.shape[1])
    ]

    df = pl.DataFrame(
        X,
        schema=pixel_columns,
        orient="row",
    ).with_columns(
        pl.Series("label", y)
    )

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
        raise FileNotFoundError(
            f"MNIST dataset not found at {MNIST_DATA_PATH}"
        )

    return download_mnist()