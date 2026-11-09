"""Tests for the tilt estimators."""

import numpy as np
import pytest

from kalman_imu.filters import (ComplementaryFilter, KalmanFilter, accel_angle,
                                rmse, run_complementary, run_kalman)
from kalman_imu.simulation import IMUConfig, simulate_imu


@pytest.mark.parametrize("deg", [-60.0, -10.0, 0.0, 25.0, 80.0])
def test_accel_angle_recovers_static_tilt(deg):
    theta = np.deg2rad(deg)
    assert accel_angle(np.sin(theta), np.cos(theta)) == pytest.approx(theta)


def test_rmse():
    assert rmse(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(np.sqrt(2.0))


def test_complementary_alpha_extremes():
    # alpha = 0 -> pure accelerometer; alpha = 1 -> pure gyro integration.
    cf = ComplementaryFilter(alpha=0.0)
    assert cf.update(gyro_rate=5.0, measured_angle=0.3, dt=0.01) == pytest.approx(0.3)
    cf = ComplementaryFilter(alpha=1.0, initial_angle=0.1)
    assert cf.update(gyro_rate=2.0, measured_angle=0.9, dt=0.01) == pytest.approx(0.12)


def test_complementary_rejects_bad_alpha():
    with pytest.raises(ValueError):
        ComplementaryFilter(alpha=1.5)


def test_kalman_converges_and_estimates_bias():
    """Sensor held still at 20 deg with a biased gyro: both states converge."""
    rng = np.random.default_rng(0)
    true_angle, true_bias, dt = np.deg2rad(20.0), np.deg2rad(2.0), 0.01
    # Small q_bias: the bias is modelled as (almost) constant.
    kf = KalmanFilter(q_bias=1e-5)
    for _ in range(3000):
        gyro = true_bias + rng.normal(0, np.deg2rad(0.5))
        meas = true_angle + rng.normal(0, np.deg2rad(3.0))
        kf.step(gyro, meas, dt)
    assert kf.angle == pytest.approx(true_angle, abs=np.deg2rad(1.0))
    assert kf.bias == pytest.approx(true_bias, abs=np.deg2rad(0.3))
    # Covariance must stay symmetric positive-definite.
    assert np.allclose(kf.P, kf.P.T)
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)


def test_filters_beat_raw_accelerometer_on_simulation():
    cfg = IMUConfig(seed=123)
    data = simulate_imu(cfg)
    acc = accel_angle(data.accel_x, data.accel_z)
    comp = run_complementary(data.gyro, acc, cfg.dt)
    kal, _ = run_kalman(data.gyro, acc, cfg.dt)

    err_acc = rmse(acc, data.true_angle)
    err_comp = rmse(comp, data.true_angle)
    err_kal = rmse(kal, data.true_angle)
    assert err_comp < err_acc
    assert err_kal < err_comp
