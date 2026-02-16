"""
Deep learning models for algorithmic trading.

This subpackage provides neural network implementations for trading
applications, including feedforward networks, RNNs, LSTMs, GRUs, and CNNs
for time series prediction and sentiment analysis.
"""

from .feedforward import (
    FeedforwardNet,
    TradingFFN
)

from .training import (
    EarlyStopping,
    LRScheduler,
    create_dataloaders,
    training_loop,
    compute_class_weights,
    set_seed
)

from .logging import (
    TrainingLogger,
    MetricsTracker,
    create_training_logger
)

try:
    from .rnn import (
        LSTMNet,
        TradingLSTM,
        StackedLSTM,
        MultivariateLSTM,
        GRUNet,
        TradingGRU
    )
    RNN_AVAILABLE = True
except ImportError:
    RNN_AVAILABLE = False

try:
    from .sentiment_rnn import (
        SentimentLSTM,
        SentimentClassifier
    )
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from .cnn import (
        Conv1DNet,
        TradingCNN
    )
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

try:
    from .cnn_ta import (
        CNNTA,
        TradingCNNTA,
        series_to_image
    )
    CNN_TA_AVAILABLE = True
except ImportError:
    CNN_TA_AVAILABLE = False

try:
    from .transfer import (
        TransferLearningModel,
        prepare_financial_images
    )
    TRANSFER_AVAILABLE = True
except ImportError:
    TRANSFER_AVAILABLE = False

try:
    from .autoencoder import (
        Autoencoder,
        DenoisingAutoencoder,
        VAE,
        ConditionalAutoencoder,
        AETrainer
    )
    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False

try:
    from .gan import (
        Generator,
        Discriminator,
        GAN,
        TimeGAN,
        SyntheticDataEvaluator
    )
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False

__all__ = [
    'FeedforwardNet',
    'TradingFFN',
    'EarlyStopping',
    'LRScheduler',
    'create_dataloaders',
    'training_loop',
    'compute_class_weights',
    'set_seed',
    'TrainingLogger',
    'MetricsTracker',
    'create_training_logger',
]

if RNN_AVAILABLE:
    __all__.extend([
        'LSTMNet',
        'TradingLSTM',
        'StackedLSTM',
        'MultivariateLSTM',
        'GRUNet',
        'TradingGRU',
    ])

if SENTIMENT_AVAILABLE:
    __all__.extend([
        'SentimentLSTM',
        'SentimentClassifier',
    ])

if CNN_AVAILABLE:
    __all__.extend([
        'Conv1DNet',
        'TradingCNN',
    ])

if CNN_TA_AVAILABLE:
    __all__.extend([
        'CNNTA',
        'TradingCNNTA',
        'series_to_image',
    ])

if TRANSFER_AVAILABLE:
    __all__.extend([
        'TransferLearningModel',
        'prepare_financial_images',
    ])

if AUTOENCODER_AVAILABLE:
    __all__.extend([
        'Autoencoder',
        'DenoisingAutoencoder',
        'VAE',
        'ConditionalAutoencoder',
        'AETrainer',
    ])

if GAN_AVAILABLE:
    __all__.extend([
        'Generator',
        'Discriminator',
        'GAN',
        'TimeGAN',
        'SyntheticDataEvaluator',
    ])
