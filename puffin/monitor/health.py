"""System health monitoring and alerting."""

import logging
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class HealthCheck:
    """Health check result."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    last_update: datetime
    latency: Optional[float] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemHealth:
    """Monitor system health and send alerts."""

    def __init__(self, alert_callback: Optional[Callable] = None):
        """
        Initialize system health monitor.

        Parameters
        ----------
        alert_callback : callable, optional
            Callback function for alerts: callback(message, level)
        """
        self.alert_callback = alert_callback
        self.logger = logging.getLogger(__name__)
        self.health_checks: Dict[str, HealthCheck] = {}

    def check_data_feed(self, provider: Any) -> Dict[str, Any]:
        """
        Check data feed provider health.

        Parameters
        ----------
        provider : object
            Data provider object (should have health check methods)

        Returns
        -------
        dict
            Health check results with keys:
            - status: 'healthy', 'degraded', 'unhealthy'
            - last_update: Timestamp of last data update
            - latency: Data feed latency in seconds
            - message: Status message

        Examples
        --------
        >>> health = SystemHealth()
        >>> # Mock provider with health check
        >>> class MockProvider:
        ...     def health_check(self):
        ...         return {
        ...             'status': 'healthy',
        ...             'last_update': datetime.now(),
        ...             'latency': 0.05
        ...         }
        >>> provider = MockProvider()
        >>> result = health.check_data_feed(provider)
        >>> result['status']
        'healthy'
        """
        try:
            # Try to get health status from provider
            if hasattr(provider, 'health_check'):
                health_data = provider.health_check()
            elif hasattr(provider, 'get_latest_timestamp'):
                # Fallback: check last update timestamp
                last_update = provider.get_latest_timestamp()
                latency = (datetime.now() - last_update).total_seconds()

                if latency < 60:
                    status = 'healthy'
                elif latency < 300:
                    status = 'degraded'
                else:
                    status = 'unhealthy'

                health_data = {
                    'status': status,
                    'last_update': last_update,
                    'latency': latency,
                    'message': f'Data latency: {latency:.1f}s'
                }
            else:
                health_data = {
                    'status': 'unknown',
                    'last_update': datetime.now(),
                    'latency': None,
                    'message': 'Provider does not support health checks'
                }

            # Store health check
            check = HealthCheck(
                status=health_data.get('status', 'unknown'),
                last_update=health_data.get('last_update', datetime.now()),
                latency=health_data.get('latency'),
                message=health_data.get('message')
            )
            self.health_checks['data_feed'] = check

            # Alert if unhealthy
            if check.status == 'unhealthy':
                self.alert(
                    f"Data feed unhealthy: {check.message}",
                    level='error'
                )
            elif check.status == 'degraded':
                self.alert(
                    f"Data feed degraded: {check.message}",
                    level='warning'
                )

            return health_data

        except Exception as e:
            error_msg = f"Data feed health check failed: {str(e)}"
            self.logger.error(error_msg)
            self.alert(error_msg, level='error')

            return {
                'status': 'unhealthy',
                'last_update': datetime.now(),
                'latency': None,
                'message': error_msg
            }

    def check_broker_connection(self, broker: Any) -> Dict[str, Any]:
        """
        Check broker connection health.

        Parameters
        ----------
        broker : object
            Broker connection object (should have health check methods)

        Returns
        -------
        dict
            Health check results with keys:
            - status: 'healthy', 'degraded', 'unhealthy'
            - last_heartbeat: Timestamp of last heartbeat
            - connected: Boolean indicating connection status
            - message: Status message

        Examples
        --------
        >>> health = SystemHealth()
        >>> # Mock broker with health check
        >>> class MockBroker:
        ...     def is_connected(self):
        ...         return True
        ...     def last_heartbeat(self):
        ...         return datetime.now()
        >>> broker = MockBroker()
        >>> result = health.check_broker_connection(broker)
        >>> result['status']
        'healthy'
        """
        try:
            # Check connection status
            if hasattr(broker, 'is_connected'):
                connected = broker.is_connected()
            else:
                connected = None

            # Check last heartbeat
            if hasattr(broker, 'last_heartbeat'):
                last_heartbeat = broker.last_heartbeat()
            else:
                last_heartbeat = datetime.now()

            # Determine health status
            if connected is False:
                status = 'unhealthy'
                message = 'Broker disconnected'
            elif connected is None:
                status = 'unknown'
                message = 'Connection status unavailable'
            else:
                # Check heartbeat timing
                heartbeat_age = (datetime.now() - last_heartbeat).total_seconds()
                if heartbeat_age < 60:
                    status = 'healthy'
                    message = f'Connected (heartbeat {heartbeat_age:.1f}s ago)'
                elif heartbeat_age < 300:
                    status = 'degraded'
                    message = f'Connection degraded (heartbeat {heartbeat_age:.1f}s ago)'
                else:
                    status = 'unhealthy'
                    message = f'Stale heartbeat ({heartbeat_age:.1f}s ago)'

            health_data = {
                'status': status,
                'last_heartbeat': last_heartbeat,
                'connected': connected,
                'message': message
            }

            # Store health check
            check = HealthCheck(
                status=status,
                last_update=datetime.now(),
                message=message,
                metadata={'connected': connected, 'last_heartbeat': last_heartbeat}
            )
            self.health_checks['broker'] = check

            # Alert if unhealthy
            if status == 'unhealthy':
                self.alert(f"Broker unhealthy: {message}", level='error')
            elif status == 'degraded':
                self.alert(f"Broker degraded: {message}", level='warning')

            return health_data

        except Exception as e:
            error_msg = f"Broker health check failed: {str(e)}"
            self.logger.error(error_msg)
            self.alert(error_msg, level='error')

            return {
                'status': 'unhealthy',
                'last_heartbeat': datetime.now(),
                'connected': False,
                'message': error_msg
            }

    def alert(self, message: str, level: str = 'warning') -> None:
        """
        Send an alert.

        Parameters
        ----------
        message : str
            Alert message
        level : str, default='warning'
            Alert level: 'info', 'warning', 'error', 'critical'

        Examples
        --------
        >>> def my_callback(msg, lvl):
        ...     print(f"{lvl.upper()}: {msg}")
        >>> health = SystemHealth(alert_callback=my_callback)
        >>> health.alert("Test alert", level='warning')
        WARNING: Test alert
        """
        # Log the alert
        log_func = getattr(self.logger, level, self.logger.warning)
        log_func(message)

        # Call alert callback if provided
        if self.alert_callback is not None:
            try:
                self.alert_callback(message, level)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.

        Returns
        -------
        dict
            Overall health status with:
            - status: Overall status ('healthy', 'degraded', 'unhealthy')
            - checks: Dictionary of individual health checks
            - timestamp: Current timestamp
        """
        if not self.health_checks:
            return {
                'status': 'unknown',
                'checks': {},
                'timestamp': datetime.now()
            }

        # Determine overall status
        statuses = [check.status for check in self.health_checks.values()]

        if 'unhealthy' in statuses:
            overall_status = 'unhealthy'
        elif 'degraded' in statuses:
            overall_status = 'degraded'
        elif all(s == 'healthy' for s in statuses):
            overall_status = 'healthy'
        else:
            overall_status = 'unknown'

        return {
            'status': overall_status,
            'checks': {
                name: {
                    'status': check.status,
                    'last_update': check.last_update,
                    'latency': check.latency,
                    'message': check.message
                }
                for name, check in self.health_checks.items()
            },
            'timestamp': datetime.now()
        }

    def reset(self):
        """Reset all health checks."""
        self.health_checks.clear()
