"""Model interpretation using SHAP values for tree ensemble models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelInterpreter:
    """Interpret tree ensemble models using SHAP values."""

    def __init__(self):
        """Initialize the model interpreter."""
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            )

    def shap_values(
        self,
        model,
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> shap.Explanation:
        """Calculate SHAP values for the model.

        Args:
            model: Trained model (XGBoost, LightGBM, CatBoost, or sklearn RandomForest).
            X: Feature matrix.
            check_additivity: Whether to check SHAP value additivity (slow).

        Returns:
            SHAP Explanation object containing values and expected value.
        """
        # Clean data
        mask = X.notna().all(axis=1)
        X_clean = X[mask]

        # Create explainer based on model type
        model_type = type(model).__name__

        if "XGB" in model_type:
            # XGBoost model
            explainer = shap.TreeExplainer(model)
        elif "LGBM" in model_type or "LightGBM" in model_type:
            # LightGBM model
            explainer = shap.TreeExplainer(model)
        elif "CatBoost" in model_type:
            # CatBoost model
            explainer = shap.TreeExplainer(model)
        elif "RandomForest" in model_type:
            # Sklearn RandomForest
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback to generic explainer
            explainer = shap.Explainer(model.predict, X_clean)

        # Calculate SHAP values
        shap_values = explainer(X_clean)

        # For multi-output models (e.g., classifiers), select positive class
        if shap_values.values.ndim == 3:
            # Take SHAP values for the last class (positive class in binary)
            shap_values = shap_values[..., -1]

        return shap_values

    def plot_summary(
        self,
        model,
        X: pd.DataFrame,
        plot_type: str = "dot",
        max_display: int = 20
    ) -> plt.Figure:
        """Create a SHAP summary plot (beeswarm or bar plot).

        Args:
            model: Trained model.
            X: Feature matrix.
            plot_type: Type of plot - 'dot' for beeswarm, 'bar' for bar chart.
            max_display: Maximum number of features to display.

        Returns:
            Matplotlib figure object.
        """
        # Calculate SHAP values
        shap_values = self.shap_values(model, X)

        # Create plot
        fig = plt.figure(figsize=(10, max(6, max_display * 0.3)))

        if plot_type == "dot":
            shap.summary_plot(
                shap_values,
                max_display=max_display,
                show=False
            )
        elif plot_type == "bar":
            shap.summary_plot(
                shap_values,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'dot' or 'bar'.")

        plt.tight_layout()
        return fig

    def plot_dependence(
        self,
        model,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: str = "auto"
    ) -> plt.Figure:
        """Create a SHAP dependence plot for a specific feature.

        Args:
            model: Trained model.
            X: Feature matrix.
            feature: Feature name to plot.
            interaction_feature: Feature to color by for interactions.
                               'auto' selects automatically.

        Returns:
            Matplotlib figure object.
        """
        # Calculate SHAP values
        shap_values = self.shap_values(model, X)

        # Create plot
        fig = plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            shap_values.values,
            X[X.notna().all(axis=1)],
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        return fig

    def plot_waterfall(
        self,
        model,
        X: pd.DataFrame,
        index: int = 0
    ) -> plt.Figure:
        """Create a SHAP waterfall plot for a single prediction.

        Args:
            model: Trained model.
            X: Feature matrix.
            index: Index of the sample to explain.

        Returns:
            Matplotlib figure object.
        """
        # Calculate SHAP values
        shap_values = self.shap_values(model, X)

        # Create plot
        fig = plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_values[index], show=False)
        plt.tight_layout()
        return fig

    def plot_force(
        self,
        model,
        X: pd.DataFrame,
        index: int = 0
    ):
        """Create a SHAP force plot for a single prediction.

        Args:
            model: Trained model.
            X: Feature matrix.
            index: Index of the sample to explain.

        Returns:
            SHAP force plot visualization object.
        """
        # Calculate SHAP values
        shap_values = self.shap_values(model, X)

        # Create force plot (returns HTML/JS visualization)
        force_plot = shap.force_plot(
            shap_values.base_values[index],
            shap_values.values[index],
            X.iloc[index]
        )

        return force_plot

    def feature_importance_comparison(
        self,
        models_dict: dict,
        X: pd.DataFrame,
        method: str = "shap"
    ) -> pd.DataFrame:
        """Compare feature importance across multiple models.

        Args:
            models_dict: Dictionary of {model_name: model} pairs.
            X: Feature matrix.
            method: Importance method - 'shap' or 'native'.

        Returns:
            DataFrame with feature importances for each model.
        """
        importance_data = {}

        for model_name, model in models_dict.items():
            if method == "shap":
                # Use SHAP values
                shap_values = self.shap_values(model, X)
                # Calculate mean absolute SHAP value for each feature
                importance = np.abs(shap_values.values).mean(axis=0)
                feature_names = X.columns.tolist()
            elif method == "native":
                # Use native feature importance
                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                    feature_names = X.columns.tolist()
                elif hasattr(model, "get_feature_importance"):
                    # CatBoost
                    importance = model.get_feature_importance()
                    feature_names = X.columns.tolist()
                else:
                    raise ValueError(f"Model {model_name} does not have feature_importances_")
            else:
                raise ValueError(f"Unknown method: {method}. Use 'shap' or 'native'.")

            importance_data[model_name] = pd.Series(importance, index=feature_names)

        # Create DataFrame
        importance_df = pd.DataFrame(importance_data)

        # Sort by mean importance across all models
        importance_df["mean_importance"] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values("mean_importance", ascending=False)

        return importance_df

    def plot_importance_comparison(
        self,
        models_dict: dict,
        X: pd.DataFrame,
        max_features: int = 20,
        method: str = "shap"
    ) -> plt.Figure:
        """Plot feature importance comparison across models.

        Args:
            models_dict: Dictionary of {model_name: model} pairs.
            X: Feature matrix.
            max_features: Maximum number of features to display.
            method: Importance method - 'shap' or 'native'.

        Returns:
            Matplotlib figure object.
        """
        # Get importance comparison
        importance_df = self.feature_importance_comparison(models_dict, X, method)

        # Remove mean_importance column for plotting
        importance_df = importance_df.drop(columns=["mean_importance"])

        # Limit to top features
        importance_df = importance_df.head(max_features)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(6, max_features * 0.3)))
        importance_df.plot(kind="barh", ax=ax)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {max_features} Feature Importance Comparison ({method.upper()})")
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return fig
