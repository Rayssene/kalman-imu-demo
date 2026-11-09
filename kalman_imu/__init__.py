"""Tilt estimation from a simulated noisy IMU.

Modules
-------
simulation : generates a ground-truth tilt trajectory and the noisy
             accelerometer / gyroscope readings a real IMU would produce.
filters    : the raw accelerometer angle, a complementary filter and a
             2-state Kalman filter (angle + gyro bias).
plotting   : comparison figures.
"""

from .filters import ComplementaryFilter, KalmanFilter, accel_angle, rmse
from .simulation import IMUConfig, IMUData, simulate_imu

__all__ = [
    "ComplementaryFilter",
    "KalmanFilter",
    "accel_angle",
    "rmse",
    "IMUConfig",
    "IMUData",
    "simulate_imu",
]
