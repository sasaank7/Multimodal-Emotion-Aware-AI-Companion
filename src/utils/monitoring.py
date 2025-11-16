"""
Monitoring and Observability Tools.
Track system metrics, performance, and health.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Callable
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """
    Collect and store application metrics.
    """

    def __init__(self, max_points: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_points: Maximum points to store per metric
        """
        self.metrics: Dict[str, deque] = {}
        self.max_points = max_points
        self.lock = threading.Lock()

        logger.info("Metrics collector initialized")

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags/labels
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )

        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.max_points)

            self.metrics[name].append(point)

    def get_metric(
        self,
        name: str,
        window_seconds: Optional[int] = None
    ) -> List[MetricPoint]:
        """
        Get metric values.

        Args:
            name: Metric name
            window_seconds: Time window (None for all)

        Returns:
            List of metric points
        """
        with self.lock:
            if name not in self.metrics:
                return []

            points = list(self.metrics[name])

            if window_seconds:
                cutoff = datetime.now() - timedelta(seconds=window_seconds)
                points = [p for p in points if p.timestamp > cutoff]

            return points

    def get_metric_stats(
        self,
        name: str,
        window_seconds: Optional[int] = None
    ) -> Dict:
        """
        Get statistics for a metric.

        Args:
            name: Metric name
            window_seconds: Time window

        Returns:
            Statistics dictionary
        """
        points = self.get_metric(name, window_seconds)

        if not points:
            return {
                'count': 0,
                'avg': 0,
                'min': 0,
                'max': 0,
                'latest': 0
            }

        values = [p.value for p in points]

        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0,
            'window_seconds': window_seconds
        }

    def get_all_metrics(self) -> Dict[str, List[Dict]]:
        """
        Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        with self.lock:
            return {
                name: [p.to_dict() for p in points]
                for name, points in self.metrics.items()
            }

    def clear_metrics(self, name: Optional[str] = None):
        """
        Clear metrics.

        Args:
            name: Specific metric to clear (None for all)
        """
        with self.lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
            else:
                self.metrics.clear()


class PerformanceMonitor:
    """
    Monitor system and application performance.
    """

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize performance monitor.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        logger.info("Performance monitor initialized")

    def start_monitoring(self, interval: float = 5.0):
        """
        Start continuous monitoring.

        Args:
            interval: Collection interval in seconds
        """
        if self.monitoring:
            logger.warning("Monitoring already started")
            return

        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                try:
                    self.collect_system_metrics()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Performance monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("Performance monitoring stopped")

    def collect_system_metrics(self):
        """Collect system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics.record_metric('system.cpu.percent', cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics.record_metric('system.memory.percent', memory.percent)
        self.metrics.record_metric('system.memory.used_mb', memory.used / (1024 ** 2))
        self.metrics.record_metric('system.memory.available_mb', memory.available / (1024 ** 2))

        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics.record_metric('system.disk.percent', disk.percent)

        # Network (if available)
        try:
            net_io = psutil.net_io_counters()
            self.metrics.record_metric('system.network.bytes_sent', net_io.bytes_sent)
            self.metrics.record_metric('system.network.bytes_recv', net_io.bytes_recv)
        except:
            pass

    def get_system_health(self) -> Dict:
        """
        Get current system health status.

        Returns:
            Health status dictionary
        """
        cpu = self.metrics.get_metric_stats('system.cpu.percent', window_seconds=60)
        memory = self.metrics.get_metric_stats('system.memory.percent', window_seconds=60)
        disk = self.metrics.get_metric_stats('system.disk.percent', window_seconds=60)

        # Determine health status
        health = 'healthy'

        if cpu.get('avg', 0) > 80 or memory.get('avg', 0) > 85:
            health = 'degraded'

        if cpu.get('avg', 0) > 95 or memory.get('avg', 0) > 95:
            health = 'unhealthy'

        return {
            'status': health,
            'cpu': cpu,
            'memory': memory,
            'disk': disk,
            'timestamp': datetime.now().isoformat()
        }


class RequestTracker:
    """
    Track API requests and performance.
    """

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize request tracker.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.active_requests: Dict[str, float] = {}
        self.lock = threading.Lock()

        logger.info("Request tracker initialized")

    def start_request(self, request_id: str):
        """
        Start tracking a request.

        Args:
            request_id: Unique request identifier
        """
        with self.lock:
            self.active_requests[request_id] = time.time()

        self.metrics.record_metric('requests.active', len(self.active_requests))

    def end_request(
        self,
        request_id: str,
        endpoint: str,
        status_code: int = 200
    ):
        """
        End request tracking and record metrics.

        Args:
            request_id: Request identifier
            endpoint: API endpoint
            status_code: HTTP status code
        """
        with self.lock:
            if request_id not in self.active_requests:
                logger.warning(f"Unknown request ID: {request_id}")
                return

            start_time = self.active_requests.pop(request_id)

        duration = time.time() - start_time

        # Record metrics
        self.metrics.record_metric(
            'requests.duration_ms',
            duration * 1000,
            tags={'endpoint': endpoint, 'status': str(status_code)}
        )

        self.metrics.record_metric('requests.total', 1, tags={'endpoint': endpoint})
        self.metrics.record_metric('requests.active', len(self.active_requests))

        if status_code >= 400:
            self.metrics.record_metric('requests.errors', 1, tags={'endpoint': endpoint})

    def get_request_stats(self, window_seconds: int = 300) -> Dict:
        """
        Get request statistics.

        Args:
            window_seconds: Time window for stats

        Returns:
            Statistics dictionary
        """
        duration_stats = self.metrics.get_metric_stats(
            'requests.duration_ms',
            window_seconds
        )

        total_points = self.metrics.get_metric('requests.total', window_seconds)
        error_points = self.metrics.get_metric('requests.errors', window_seconds)

        total_requests = len(total_points)
        total_errors = len(error_points)

        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        return {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate_percent': error_rate,
            'avg_response_time_ms': duration_stats.get('avg', 0),
            'min_response_time_ms': duration_stats.get('min', 0),
            'max_response_time_ms': duration_stats.get('max', 0),
            'active_requests': len(self.active_requests),
            'window_seconds': window_seconds
        }


class AlertManager:
    """
    Manage alerts based on metrics.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize alert manager.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector
        self.alert_rules: List[Dict] = []
        self.active_alerts: List[Dict] = []
        self.alert_callbacks: List[Callable] = []

        logger.info("Alert manager initialized")

    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: Callable[[float], bool],
        severity: str = 'warning',
        message: str = ''
    ):
        """
        Add an alert rule.

        Args:
            name: Alert name
            metric_name: Metric to monitor
            condition: Function that returns True when alert should trigger
            severity: Alert severity (info, warning, critical)
            message: Alert message
        """
        rule = {
            'name': name,
            'metric_name': metric_name,
            'condition': condition,
            'severity': severity,
            'message': message
        }

        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")

    def check_alerts(self):
        """Check all alert rules."""
        for rule in self.alert_rules:
            metric_stats = self.metrics.get_metric_stats(
                rule['metric_name'],
                window_seconds=60
            )

            latest_value = metric_stats.get('latest', 0)

            if rule['condition'](latest_value):
                self._trigger_alert(rule, latest_value)

    def _trigger_alert(self, rule: Dict, value: float):
        """
        Trigger an alert.

        Args:
            rule: Alert rule
            value: Metric value that triggered alert
        """
        alert = {
            'name': rule['name'],
            'severity': rule['severity'],
            'message': rule['message'] or f"{rule['name']}: {value}",
            'metric_value': value,
            'timestamp': datetime.now().isoformat()
        }

        self.active_alerts.append(alert)

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Alert triggered: {alert['name']} - {alert['message']}")

    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """
        Add callback to be called when alerts trigger.

        Args:
            callback: Callback function
        """
        self.alert_callbacks.append(callback)

    def get_active_alerts(self) -> List[Dict]:
        """Get active alerts."""
        return self.active_alerts.copy()

    def clear_alerts(self):
        """Clear all active alerts."""
        self.active_alerts.clear()


# Global instances
_metrics_collector = MetricsCollector()
_performance_monitor = PerformanceMonitor(_metrics_collector)
_request_tracker = RequestTracker(_metrics_collector)


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return _performance_monitor


def get_request_tracker() -> RequestTracker:
    """Get global request tracker."""
    return _request_tracker
