"""
Demonstration of linear models for trading.

This script shows how to use OLS, Ridge, Lasso, and logistic regression
for return prediction and direction classification.
"""

import numpy as np
import pandas as pd
from puffin.models.linear import OLSModel, RidgeModel, LassoModel, DirectionClassifier
from puffin.models.factor_models import FamaFrenchModel, fama_macbeth


def demo_basic_regression():
    """Demonstrate basic regression models."""
    print("=" * 70)
    print("DEMO 1: Basic Regression Models")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 8

    # Create features with some being more important
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([2.0, -1.5, 0.8, 0.0, 0.0, 1.2, -0.5, 0.0])
    y = 5.0 + X @ true_coef + np.random.randn(n_samples) * 0.5

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='returns')

    # Split data
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X_df[:split_idx], X_df[split_idx:]
    y_train, y_test = y_series[:split_idx], y_series[split_idx:]

    # OLS Model
    print("\n1. OLS Regression")
    print("-" * 70)
    ols = OLSModel(add_constant=True)
    ols.fit(X_train, y_train)

    print(f"R-squared: {ols.r_squared:.4f}")
    print("\nTop 3 Features by Coefficient:")
    coef = ols.coefficients.drop('const')
    print(coef.abs().sort_values(ascending=False).head(3))

    # Ridge Model
    print("\n2. Ridge Regression")
    print("-" * 70)
    ridge = RidgeModel(alphas=np.logspace(-3, 3, 30), cv=5)
    ridge.fit(X_train, y_train)

    print(f"Selected alpha: {ridge.alpha:.4f}")
    print("\nTop 3 Features by Importance:")
    print(ridge.feature_importance().head(3))

    # Lasso Model
    print("\n3. Lasso Regression")
    print("-" * 70)
    lasso = LassoModel(alphas=np.logspace(-4, 0, 30), cv=5)
    lasso.fit(X_train, y_train)

    print(f"Selected alpha: {lasso.alpha:.6f}")
    print(f"Selected Features: {len(lasso.selected_features)}/{n_features}")
    print("Selected Features:", lasso.selected_features)

    # Compare predictions
    print("\n4. Model Comparison")
    print("-" * 70)
    from sklearn.metrics import mean_squared_error, r2_score

    for name, model in [('OLS', ols), ('Ridge', ridge), ('Lasso', lasso)]:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name:10s} - MSE: {mse:.4f}, R²: {r2:.4f}")


def demo_direction_classification():
    """Demonstrate direction classification."""
    print("\n" + "=" * 70)
    print("DEMO 2: Direction Classification")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 800
    n_features = 6

    X = np.random.randn(n_samples, n_features)
    # Direction based on linear combination
    z = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] + 0.1 * X[:, 3]
    y = (z + np.random.randn(n_samples) * 0.3 > 0).astype(int)

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='direction')

    # Split data
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X_df[:split_idx], X_df[split_idx:]
    y_train, y_test = y_series[:split_idx], y_series[split_idx:]

    # Fit classifier
    classifier = DirectionClassifier(class_weight='balanced')
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Metrics
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

    print("\nFeature Importance:")
    print(classifier.feature_importance().head(4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def demo_factor_models():
    """Demonstrate factor models."""
    print("\n" + "=" * 70)
    print("DEMO 3: Factor Models (CAPM & Fama-French)")
    print("=" * 70)

    # Generate synthetic returns
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)

    # Asset returns with known factor exposures
    beta_mkt = 1.2
    beta_smb = 0.3
    beta_hml = -0.2

    # Market factor
    mkt_rf = np.random.normal(0.0004, 0.01, n_days)

    # Generate asset returns
    returns = (0.0001 +  # alpha
              beta_mkt * mkt_rf +
              np.random.normal(0, 0.005, n_days))  # idiosyncratic

    returns = pd.Series(returns, index=dates, name='returns')

    # Initialize model
    ff_model = FamaFrenchModel()

    # Fit CAPM
    print("\n1. CAPM Results")
    print("-" * 70)
    capm = ff_model.fit_capm(returns, start='2020-01-01', end='2023-12-31')

    print(f"Alpha: {capm['alpha']:.6f} (p={capm['alpha_pvalue']:.4f})")
    print(f"Beta:  {capm['beta']:.4f} (p={capm['beta_pvalue']:.4f})")
    print(f"R²:    {capm['r_squared']:.4f}")

    # Fit 3-factor
    print("\n2. Fama-French 3-Factor Results")
    print("-" * 70)
    ff3 = ff_model.fit_three_factor(returns, start='2020-01-01', end='2023-12-31')

    print(f"Alpha:      {ff3['alpha']:.6f} (p={ff3['pvalues']['alpha']:.4f})")
    print(f"Market:     {ff3['beta_mkt']:.4f} (p={ff3['pvalues']['Mkt-RF']:.4f})")
    print(f"SMB:        {ff3['beta_smb']:.4f} (p={ff3['pvalues']['SMB']:.4f})")
    print(f"HML:        {ff3['beta_hml']:.4f} (p={ff3['pvalues']['HML']:.4f})")
    print(f"R²:         {ff3['r_squared']:.4f}")

    # Fit 5-factor
    print("\n3. Fama-French 5-Factor Results")
    print("-" * 70)
    ff5 = ff_model.fit_five_factor(returns, start='2020-01-01', end='2023-12-31')

    print(f"Alpha:      {ff5['alpha']:.6f}")
    print("\nFactor Loadings:")
    for factor, beta in ff5['betas'].items():
        pval = ff5['pvalues'][factor]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {factor:10s}: {beta:7.4f} {sig}")
    print(f"\nR²:         {ff5['r_squared']:.4f}")


def demo_fama_macbeth():
    """Demonstrate Fama-MacBeth regression."""
    print("\n" + "=" * 70)
    print("DEMO 4: Fama-MacBeth Regression")
    print("=" * 70)

    # Generate synthetic panel data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
    assets = [f'Asset_{i}' for i in range(30)]

    # True risk premiums
    lambda_0 = 0.004  # zero-beta rate
    lambda_mkt = 0.006  # market premium
    lambda_smb = 0.002  # size premium
    lambda_hml = 0.003  # value premium

    data = []
    for date in dates:
        for asset in assets:
            # Random factor exposures
            beta_mkt = np.random.uniform(0.5, 1.5)
            beta_smb = np.random.uniform(-0.5, 0.5)
            beta_hml = np.random.uniform(-0.5, 0.5)

            # Returns based on factor model
            ret = (lambda_0 +
                   lambda_mkt * beta_mkt +
                   lambda_smb * beta_smb +
                   lambda_hml * beta_hml +
                   np.random.randn() * 0.02)

            data.append({
                'date': date,
                'asset': asset,
                'returns': ret,
                'beta_mkt': beta_mkt,
                'beta_smb': beta_smb,
                'beta_hml': beta_hml,
            })

    panel_df = pd.DataFrame(data)

    # Run Fama-MacBeth
    print("\nRunning Fama-MacBeth procedure...")
    fm_results = fama_macbeth(
        panel_data=panel_df,
        factors=['beta_mkt', 'beta_smb', 'beta_hml'],
        return_col='returns',
        time_col='date',
        asset_col='asset'
    )

    print("\nRisk Premiums:")
    print("-" * 70)
    for factor, premium in fm_results['risk_premiums'].items():
        t_stat = fm_results['t_stats'][factor]
        sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.65 else ""
        print(f"{factor:15s}: {premium:8.6f} (t={t_stat:6.2f}) {sig}")

    print(f"\nAverage R²: {fm_results['r_squared']:.4f}")
    print(f"Periods:    {fm_results['n_periods']}")

    print("\nTrue vs. Estimated Premiums:")
    print("-" * 70)
    true_premiums = {
        'lambda_0': lambda_0,
        'beta_mkt': lambda_mkt,
        'beta_smb': lambda_smb,
        'beta_hml': lambda_hml,
    }
    for factor, true_val in true_premiums.items():
        est_val = fm_results['risk_premiums'][factor]
        error = est_val - true_val
        print(f"{factor:15s}: True={true_val:.6f}, Est={est_val:.6f}, Error={error:.6f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PUFFIN LINEAR MODELS DEMONSTRATION")
    print("=" * 70)

    demo_basic_regression()
    demo_direction_classification()
    demo_factor_models()
    demo_fama_macbeth()

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
