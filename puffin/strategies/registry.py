"""Strategy registry for dynamic discovery and instantiation."""

from puffin.strategies.base import Strategy

_registry: dict[str, type[Strategy]] = {}


def register(name: str | None = None):
    """Decorator to register a strategy class."""
    def decorator(cls):
        key = name or cls.__name__
        _registry[key] = cls
        return cls
    return decorator


def get_strategy(name: str, **kwargs) -> Strategy:
    """Instantiate a registered strategy by name."""
    if name not in _registry:
        raise KeyError(f"Strategy '{name}' not found. Available: {list(_registry.keys())}")
    return _registry[name](**kwargs)


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return list(_registry.keys())


def register_defaults():
    """Register all built-in strategies."""
    from puffin.strategies.momentum import MomentumStrategy
    from puffin.strategies.mean_reversion import MeanReversionStrategy
    from puffin.strategies.stat_arb import StatArbStrategy
    from puffin.strategies.market_making import MarketMakingStrategy

    _registry["momentum"] = MomentumStrategy
    _registry["mean_reversion"] = MeanReversionStrategy
    _registry["stat_arb"] = StatArbStrategy
    _registry["market_making"] = MarketMakingStrategy


# Auto-register defaults on import
register_defaults()
