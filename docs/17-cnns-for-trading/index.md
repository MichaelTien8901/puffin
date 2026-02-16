---
layout: default
title: "Part 17: CNNs for Trading"
nav_order: 18
has_children: true
---

# Part 17: CNNs for Trading

Apply convolutional neural networks to financial time series and alternative data.

## Overview

Convolutional Neural Networks (CNNs) are powerful deep learning models that excel at pattern recognition. While originally developed for computer vision, CNNs have proven surprisingly effective for financial time series analysis. This section explores three innovative approaches:

1. **1D CNNs**: Apply temporal convolutions for autoregressive prediction
2. **CNN-TA**: Treat time series as 2D images with technical indicators
3. **Transfer Learning**: Leverage pretrained models for financial forecasting

## Chapters

1. [CNNs for Trading](01-cnns-trading.md) - Comprehensive guide to using CNNs in algorithmic trading, including practical implementations and trading strategies

## Key Topics Covered

- 1D convolutional networks for time series forecasting
- Converting OHLCV data to image representations
- Technical indicator visualization as 2D heatmaps
- Transfer learning with pretrained models (ResNet, VGG, MobileNet)
- Fine-tuning strategies for financial data
- Complete trading strategy implementations
- Performance evaluation and best practices

## Code Examples

All implementations are available in the `puffin.deep` package:
- `puffin.deep.cnn`: 1D CNN for time series
- `puffin.deep.cnn_ta`: 2D CNN with technical analysis
- `puffin.deep.transfer`: Transfer learning models

## Prerequisites

- Understanding of neural networks and deep learning
- Familiarity with PyTorch
- Knowledge of technical analysis indicators
- Basic understanding of image processing concepts
