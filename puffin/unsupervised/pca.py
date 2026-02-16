"""PCA and eigenportfolio analysis for trading."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from typing import Optional


class MarketPCA:
    """Principal Component Analysis for market returns.

    Attributes:
        explained_variance_ratio: Proportion of variance explained by each component.
        components: Principal components (eigenvectors).
        n_components_95: Number of components needed for 95% variance.
    """

    def __init__(self, n_components: Optional[int] = None):
        """Initialize MarketPCA.

        Args:
            n_components: Number of components to keep. None = all components.
        """
        self.n_components = n_components
        self._pca: Optional[SklearnPCA] = None
        self._feature_names: Optional[list] = None

    def fit(self, returns: pd.DataFrame) -> "MarketPCA":
        """Fit PCA on returns data.

        Args:
            returns: DataFrame of asset returns (rows=dates, cols=assets).

        Returns:
            Self for method chaining.
        """
        returns_clean = returns.dropna()
        self._feature_names = returns_clean.columns.tolist()

        self._pca = SklearnPCA(n_components=self.n_components)
        self._pca.fit(returns_clean)

        return self

    def transform(self, returns: pd.DataFrame) -> np.ndarray:
        """Transform returns to principal component space.

        Args:
            returns: DataFrame of asset returns.

        Returns:
            Array of transformed data (n_samples, n_components).
        """
        if self._pca is None:
            raise ValueError("Must call fit() before transform()")

        returns_clean = returns.dropna()
        return self._pca.transform(returns_clean)

    def fit_transform(self, returns: pd.DataFrame) -> np.ndarray:
        """Fit PCA and transform in one step.

        Args:
            returns: DataFrame of asset returns.

        Returns:
            Array of transformed data.
        """
        return self.fit(returns).transform(returns)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get proportion of variance explained by each component."""
        if self._pca is None:
            raise ValueError("Must call fit() first")
        return self._pca.explained_variance_ratio_

    @property
    def components(self) -> np.ndarray:
        """Get principal components (eigenvectors)."""
        if self._pca is None:
            raise ValueError("Must call fit() first")
        return self._pca.components_

    @property
    def n_components_95(self) -> int:
        """Get number of components needed for 95% variance."""
        if self._pca is None:
            raise ValueError("Must call fit() first")

        cumsum = np.cumsum(self.explained_variance_ratio)
        return int(np.argmax(cumsum >= 0.95) + 1)

    def eigenportfolios(self, returns: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Extract top N eigenportfolios (portfolio weights from PCA).

        Args:
            returns: DataFrame of asset returns.
            n: Number of eigenportfolios to return.

        Returns:
            DataFrame of portfolio weights (n_components x n_assets).
        """
        if self._pca is None:
            self.fit(returns)

        n_components = min(n, self._pca.n_components_)
        weights = self.components[:n_components]

        # Normalize to sum to 1 (long-only portfolio convention)
        weights_abs = np.abs(weights)
        weights_norm = weights_abs / weights_abs.sum(axis=1, keepdims=True)

        return pd.DataFrame(
            weights_norm,
            columns=self._feature_names,
            index=[f"PC{i+1}" for i in range(n_components)]
        )

    def reconstruct(self, returns: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """Reconstruct returns using first n_components.

        Args:
            returns: DataFrame of asset returns.
            n_components: Number of components to use for reconstruction.

        Returns:
            DataFrame of reconstructed returns.
        """
        if self._pca is None:
            self.fit(returns)

        returns_clean = returns.dropna()

        # Transform and inverse transform with limited components
        pca_limited = SklearnPCA(n_components=n_components)
        pca_limited.fit(returns_clean)

        transformed = pca_limited.transform(returns_clean)
        reconstructed = pca_limited.inverse_transform(transformed)

        return pd.DataFrame(
            reconstructed,
            index=returns_clean.index,
            columns=returns_clean.columns
        )

    def explained_variance_plot(self) -> pd.DataFrame:
        """Get data for plotting explained variance.

        Returns:
            DataFrame with component number, individual variance, and cumulative variance.
        """
        if self._pca is None:
            raise ValueError("Must call fit() first")

        cumsum = np.cumsum(self.explained_variance_ratio)

        return pd.DataFrame({
            "component": range(1, len(self.explained_variance_ratio) + 1),
            "variance_explained": self.explained_variance_ratio,
            "cumulative_variance": cumsum
        })
