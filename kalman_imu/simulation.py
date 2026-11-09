"""Simulation of a noisy 2-axis accelerometer and a 1-axis gyroscope.

The sensor is rotated about its y-axis by a tilt angle ``theta(t)``.

* The **gyroscope** measures the angular rate ``d(theta)/dt`` but is corrupted
  by a constant bias (which makes pure integration drift) and white noise.
* The **accelerometer** measures the projection of gravity onto the body
  axes::

      a_x = g * sin(theta)
      a_z = g * cos(theta)

  plus white noise and short bursts of *linear* acceleration (vibration /
  motion), which an accelerometer cannot distinguish from gravity.

All angles are in radians and all rates in rad/s internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81  # m/s^2


@dataclass
class IMUConfig:
    """Parameters of the simulated trajectory and sensor errors."""

    dt: float = 0.01                            # sample period [s] (100 Hz)
    duration: float = 20.0                      # total simulated time [s]
    gyro_bias: float = np.deg2rad(3.0)          # constant gyro bias [rad/s]
    gyro_noise_std: float = np.deg2rad(1.0)     # gyro white noise [rad/s]
    accel_noise_std: float = 0.5                # accel white noise [m/s^2]
    disturbance_amplitude: float = 2.0          # linear-accel bursts [m/s^2]
    seed: int | None = 42                       # RNG seed (None = random)


@dataclass
class IMUData:
    """Time series produced by :func:`simulate_imu`."""

    t: np.ndarray           # time stamps [s]
    true_angle: np.ndarray  # ground-truth tilt [rad]
    true_rate: np.ndarray   # ground-truth angular rate [rad/s]
    gyro: np.ndarray        # measured angular rate [rad/s]
    accel_x: np.ndarray     # measured acceleration along x [m/s^2]
    accel_z: np.ndarray     # measured acceleration along z [m/s^2]


def true_motion(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth tilt angle and its analytic derivative.

    A sum of two sinusoids gives slow large swings plus faster small
    oscillations, which exercises both the gyro and the accelerometer paths.
    """
    a1, w1 = np.deg2rad(30.0), 0.5   # 30 deg amplitude, slow
    a2, w2 = np.deg2rad(10.0), 2.0   # 10 deg amplitude, faster
    angle = a1 * np.sin(w1 * t) + a2 * np.sin(w2 * t)
    rate = a1 * w1 * np.cos(w1 * t) + a2 * w2 * np.cos(w2 * t)
    return angle, rate


def _linear_disturbance(t: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Short vibration bursts of linear acceleration (non-gravitational).

    Three 1-second windows of a 5 Hz oscillation with random phase. These
    fool the accelerometer-only angle, which is exactly what the fusion
    filters are meant to reject.
    """
    disturbance = np.zeros_like(t)
    for start in (4.0, 9.0, 15.0):
        window = (t >= start) & (t < start + 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        disturbance[window] = amplitude * np.sin(2 * np.pi * 5.0 * t[window] + phase)
    return disturbance


def simulate_imu(config: IMUConfig | None = None) -> IMUData:
    """Generate a ground-truth trajectory and the matching noisy IMU readings."""
    cfg = config or IMUConfig()
    rng = np.random.default_rng(cfg.seed)

    n = int(round(cfg.duration / cfg.dt))
    t = np.arange(n) * cfg.dt
    angle, rate = true_motion(t)

    # Gyroscope: true rate + constant bias + white noise.
    gyro = rate + cfg.gyro_bias + rng.normal(0.0, cfg.gyro_noise_std, n)

    # Accelerometer: gravity projection + vibration (along x) + white noise.
    disturbance = _linear_disturbance(t, cfg.disturbance_amplitude, rng)
    accel_x = GRAVITY * np.sin(angle) + disturbance + rng.normal(0.0, cfg.accel_noise_std, n)
    accel_z = GRAVITY * np.cos(angle) + rng.normal(0.0, cfg.accel_noise_std, n)

    return IMUData(t=t, true_angle=angle, true_rate=rate, gyro=gyro,
                   accel_x=accel_x, accel_z=accel_z)
