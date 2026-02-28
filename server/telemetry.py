"""
Telemetry stub for tracking signals and events.
"""

class Telemetry:
    """Simple telemetry tracker."""
    
    def track_signal(self, source: str, priority: float = 1.0):
        """Track a signal event."""
        pass
    
    def track_event(self, event_name: str, **kwargs):
        """Track a generic event."""
        pass


# Singleton instance
telemetry = Telemetry()
