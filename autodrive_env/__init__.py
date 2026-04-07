"""AutoDrive Gym package."""

from .client import AutoDriveClient
from .models import AutoDriveAction, AutoDriveObservation, AutoDriveState

try:
    from .server.autodrive_gym_environment import AutoDriveGymEnvironment as AutoDriveEnv
except Exception:  # pragma: no cover - allows lightweight imports without full server deps
    AutoDriveEnv = None

__all__ = [
    "AutoDriveAction",
    "AutoDriveClient",
    "AutoDriveEnv",
    "AutoDriveObservation",
    "AutoDriveState",
]
