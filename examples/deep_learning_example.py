"""
Example: Deep Learning for Stock Return Prediction

This example demonstrates:
1. Building a feedforward neural network
2. Training with early stopping and LR scheduling
3. TensorBoard logging
4. Model evaluation and saving
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

from puffin.deep import (
    TradingFFN,
    FeedforwardNet,
    EarlyStopping,
    LRScheduler,
    create_dataloaders,
    training_loop,
    set_seed
)
from puffin.deep.logging import TrainingLogger, create_training_logger


def generate_synthetic_market_data(n_samples=2000, n_features=20):
    """Generate synthetic market data for demonstration."""
    np.random.seed(42)

    # Generate features (technical indicators, etc.)
    X = np.random.randn(n_samples, n_features)

    # Generate target: returns with some predictable patterns
    # Pattern 1: Momentum effect
    momentum = X[:, 0] * 0.5
    # Pattern 2: Mean reversion
    mean_reversion = -X[:, 1] * 0.3
    # Pattern 3: Volume effect
    volume_effect = X[:, 2] * 0.2
    # Add noise
    noise = np.random.randn(n_samples) * 0.5

    y = momentum + mean_reversion + volume_effect + noise

    return X, y


def example_1_basic_training():
    """Example 1: Basic feedforward network training."""
    print("=" * 60)
    print("Example 1: Basic Training")
    print("=" * 60)

    # Set random seed
    set_seed(42)

    # Generate data
    X, y = generate_synthetic_market_data(n_samples=1000, n_features=20)

    # Train/test split (time-series aware)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Create model
    model = TradingFFN(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout=0.3,
        activation='relu'
    )

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        lr=0.001,
        batch_size=64,
        validation_split=0.2,
        verbose=True
    )

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Evaluate
    from sklearn.metrics import mean_squared_error, r2_score

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print("\nResults:")
    print(f"Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")

    # Save model
    save_path = "models/basic_ffn"
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    return model, history


def example_2_advanced_training():
    """Example 2: Training with early stopping and LR scheduling."""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Training with Callbacks")
    print("=" * 60)

    # Set random seed
    set_seed(42)

    # Generate data
    X, y = generate_synthetic_market_data(n_samples=1000, n_features=20)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        batch_size=64,
        val_split=0.2,
        random_seed=42
    )

    # Create model
    model = FeedforwardNet(
        input_dim=20,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        dropout=0.4,
        activation='relu'
    )

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Create callbacks
    early_stop = EarlyStopping(
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=True
    )

    scheduler = LRScheduler(
        optimizer,
        schedule_type='cosine',
        T_max=100,
        eta_min=0.00001
    )

    def early_stop_callback(epoch, train_loss, val_loss, model):
        return early_stop(val_loss, model)

    def scheduler_callback(epoch, train_loss, val_loss, model):
        scheduler.step()
        return False

    # Train with callbacks
    print("\nTraining with early stopping and LR scheduling...")
    history = training_loop(
        model, train_loader, val_loader,
        epochs=100,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[early_stop_callback, scheduler_callback],
        device=device,
        verbose=True
    )

    print(f"\nTraining stopped at epoch {len(history['train_loss'])}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")

    return model, history


def example_3_tensorboard_logging():
    """Example 3: Training with TensorBoard logging."""
    print("\n" + "=" * 60)
    print("Example 3: Training with TensorBoard Logging")
    print("=" * 60)

    # Set random seed
    set_seed(42)

    # Generate data
    X, y = generate_synthetic_market_data(n_samples=1000, n_features=20)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        batch_size=64,
        val_split=0.2,
        random_seed=42
    )

    # Create model
    model = FeedforwardNet(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout=0.3,
        activation='relu'
    )

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Create logger
    hparams = {
        'hidden_dims': '[64, 32]',
        'dropout': 0.3,
        'lr': 0.001,
        'batch_size': 64,
        'activation': 'relu'
    }
    logger = create_training_logger(
        model_name='return_predictor',
        hparams=hparams,
        log_dir='runs'
    )

    # Log model graph
    sample_input = torch.randn(1, 20).to(device)
    logger.log_model_graph(model, sample_input)

    print("\nTraining with TensorBoard logging...")

    # Custom training loop with logging
    best_val_loss = float('inf')
    for epoch in range(50):
        # Training phase
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # Log metrics
        logger.log_scalars(
            epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            learning_rate=optimizer.param_groups[0]['lr']
        )

        # Log weights and gradients every 10 epochs
        if epoch % 10 == 0:
            logger.log_weights(epoch, model)
            logger.log_gradients(epoch, model)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50 - Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    # Log final metrics
    logger.log_hyperparameters(
        hparams,
        {'best_val_loss': best_val_loss}
    )

    logger.close()

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
    print("View TensorBoard: tensorboard --logdir=runs")

    return model


def example_4_model_comparison():
    """Example 4: Comparing different architectures."""
    print("\n" + "=" * 60)
    print("Example 4: Architecture Comparison")
    print("=" * 60)

    # Set random seed
    set_seed(42)

    # Generate data
    X, y = generate_synthetic_market_data(n_samples=1000, n_features=20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Define architectures to compare
    architectures = [
        {'name': 'Shallow', 'hidden_dims': [32], 'dropout': 0.2},
        {'name': 'Medium', 'hidden_dims': [64, 32], 'dropout': 0.3},
        {'name': 'Deep', 'hidden_dims': [128, 64, 32], 'dropout': 0.4},
        {'name': 'Wide', 'hidden_dims': [128, 128], 'dropout': 0.3},
    ]

    results = []

    for arch in architectures:
        print(f"\nTraining {arch['name']} network: {arch['hidden_dims']}")

        model = TradingFFN(
            input_dim=20,
            hidden_dims=arch['hidden_dims'],
            output_dim=1,
            dropout=arch['dropout'],
            activation='relu'
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,
            lr=0.001,
            batch_size=64,
            validation_split=0.2,
            verbose=False
        )

        # Evaluate
        test_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)

        results.append({
            'Architecture': arch['name'],
            'Hidden Dims': str(arch['hidden_dims']),
            'Test MSE': test_mse,
            'Test R²': test_r2,
            'Final Train Loss': history['train_loss'][-1],
            'Final Val Loss': history['val_loss'][-1]
        })

    # Display results
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Architecture Comparison Results")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Find best model
    best_idx = results_df['Test MSE'].idxmin()
    print(f"\nBest architecture: {results_df.iloc[best_idx]['Architecture']}")

    return results_df


if __name__ == '__main__':
    print("\nDeep Learning for Trading - Examples\n")

    # Run examples
    try:
        # Example 1: Basic training
        model1, history1 = example_1_basic_training()

        # Example 2: Advanced training with callbacks
        model2, history2 = example_2_advanced_training()

        # Example 3: TensorBoard logging
        model3 = example_3_tensorboard_logging()

        # Example 4: Model comparison
        results = example_4_model_comparison()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
