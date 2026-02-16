# Time Series Models Tests

This directory contains tests for the time series models in the Puffin library.

## Test Files

- `test_ts_diagnostics.py`: Tests for time series diagnostics (stationarity tests, decomposition, ACF/PACF)
- `test_arima.py`: Tests for ARIMA models
- `test_cointegration.py`: Tests for cointegration analysis and pairs trading

## Running Tests

To run all time series model tests:

```bash
pytest tests/models/
```

To run specific test files:

```bash
# Time series diagnostics
pytest tests/models/test_ts_diagnostics.py -v

# ARIMA models
pytest tests/models/test_arima.py -v

# Cointegration tests
pytest tests/models/test_cointegration.py -v
```

To run specific test classes:

```bash
# Test stationarity functions
pytest tests/models/test_ts_diagnostics.py::TestStationarity -v

# Test ARIMA order selection
pytest tests/models/test_arima.py::TestOrderSelection -v

# Test cointegration
pytest tests/models/test_cointegration.py::TestEngleGrangerTest -v
```

## Test Coverage

Run with coverage report:

```bash
pytest tests/models/ --cov=puffin.models --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html
```

## Dependencies

Make sure you have the ML dependencies installed:

```bash
pip install -e ".[ml]"
```

This includes:
- statsmodels (for ARIMA, VAR, stationarity tests)
- arch (for GARCH models)
- numpy, pandas, scipy
- matplotlib (for plotting)

## Test Data

The tests use synthetic data generated with specific properties:
- White noise (stationary)
- Random walks (non-stationary)
- AR(1) processes
- MA(1) processes
- Cointegrated pairs
- Seasonal time series

This ensures tests are reproducible and don't depend on external data sources.
