"""Tilt-angle estimators: raw accelerometer, complementary filter, Kalman filter.

All functions and classes work in radians.
"""

from __future__ import annotations

import numpy as np


def accel_angle(accel_x: np.ndarray | float, accel_z: np.ndarray | float) -> np.ndarray | float:
    """Tilt angle implied by the gravity direction: ``theta = atan2(a_x, a_z)``.

    Unbiased on average but noisy, and wrong whenever the sensor undergoes
    linear acceleration (gravity and motion look identical to it).
    """
    return np.arctan2(accel_x, accel_z)


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error between two equally sized arrays."""
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


class ComplementaryFilter:
    """Blend of integrated gyro (high-pass) and accelerometer angle (low-pass).

    ``angle_k = alpha * (angle_{k-1} + gyro_k * dt) + (1 - alpha) * accel_angle_k``

    ``alpha`` close to 1 trusts the gyro for fast changes while the small
    accelerometer contribution slowly pulls the estimate back, cancelling
    drift. The time constant is ``tau = alpha * dt / (1 - alpha)``.
    """

    def __init__(self, alpha: float = 0.98, initial_angle: float = 0.0) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.angle = initial_angle

    def update(self, gyro_rate: float, measured_angle: float, dt: float) -> float:
        """Advance one sample and return the new angle estimate."""
        gyro_prediction = self.angle + gyro_rate * dt
        self.angle = self.alpha * gyro_prediction + (1.0 - self.alpha) * measured_angle
        return self.angle


class KalmanFilter:
    """Two-state Kalman filter estimating tilt angle and gyro bias.

    State ``x = [angle, bias]^T``. The gyro reading is used as a control
    input; the accelerometer angle is the measurement::

        predict:  x = F x + B u           F = [[1, -dt], [0, 1]],  B = [dt, 0]^T
                  P = F P F^T + Q         Q = diag(q_angle, q_bias) * dt
        update:   y = z - H x             H = [1, 0]
                  S = H P H^T + R
                  K = P H^T / S
                  x = x + K y
                  P = (I - K H) P

    Parameters
    ----------
    q_angle : process noise spectral density on the angle [rad^2/s]
    q_bias  : process noise spectral density on the bias (random walk) [rad^2/s^3]
    r_measure : variance of the accelerometer angle measurement [rad^2]
    """

    def __init__(self, q_angle: float = 0.001, q_bias: float = 0.003,
                 r_measure: float = 0.03, initial_angle: float = 0.0) -> None:
        self.q_angle = q_angle
        self.q_bias = q_bias
        self.r_measure = r_measure
        self.x = np.array([initial_angle, 0.0])  # [angle, gyro bias]
        self.P = np.eye(2)                         # initial uncertainty
        self._H = np.array([[1.0, 0.0]])

    @property
    def angle(self) -> float:
        return float(self.x[0])

    @property
    def bias(self) -> float:
        return float(self.x[1])

    def predict(self, gyro_rate: float, dt: float) -> None:
        """Propagate the state with the (bias-corrected) gyro rate."""
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        B = np.array([dt, 0.0])
        Q = np.diag([self.q_angle, self.q_bias]) * dt
        self.x = F @ self.x + B * gyro_rate
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_angle: float) -> None:
        """Correct the state with an accelerometer angle measurement."""
        H = self._H
        y = measured_angle - (H @ self.x)[0]            # innovation
        S = (H @ self.P @ H.T)[0, 0] + self.r_measure   # innovation variance
        K = (self.P @ H.T)[:, 0] / S                    # Kalman gain (2,)
        self.x = self.x + K * y
        # Joseph form keeps P symmetric positive-definite numerically.
        I_KH = np.eye(2) - np.outer(K, H[0])
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.r_measure

    def step(self, gyro_rate: float, measured_angle: float, dt: float) -> float:
        """One predict + update cycle; returns the new angle estimate."""
        self.predict(gyro_rate, dt)
        self.update(measured_angle)
        return self.angle


def run_complementary(gyro: np.ndarray, measured: np.ndarray, dt: float,
                      alpha: float = 0.98) -> np.ndarray:
    """Run a :class:`ComplementaryFilter` over whole arrays."""
    cf = ComplementaryFilter(alpha=alpha, initial_angle=float(measured[0]))
    return np.array([cf.update(g, m, dt) for g, m in zip(gyro, measured)])


def run_kalman(gyro: np.ndarray, measured: np.ndarray, dt: float,
               **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Run a :class:`KalmanFilter` over whole arrays.

    Returns ``(angles, biases)``.
    """
    kf = KalmanFilter(initial_angle=float(measured[0]), **kwargs)
    angles = np.empty(len(gyro))
    biases = np.empty(len(gyro))
    for i, (g, m) in enumerate(zip(gyro, measured)):
        angles[i] = kf.step(g, m, dt)
        biases[i] = kf.bias
    return angles, biases
