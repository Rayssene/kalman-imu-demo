"""Tests for the IMU simulation."""

import numpy as np

from kalman_imu.simulation import GRAVITY, IMUConfig, simulate_imu


def test_shapes_and_time_axis():
    cfg = IMUConfig(dt=0.01, duration=5.0)
    data = simulate_imu(cfg)
    assert len(data.t) == 500
    for arr in (data.true_angle, data.true_rate, data.gyro, data.accel_x, data.accel_z):
        assert arr.shape == data.t.shape
    assert np.allclose(np.diff(data.t), cfg.dt)


def test_same_seed_is_reproducible():
    a = simulate_imu(IMUConfig(seed=7))
    b = simulate_imu(IMUConfig(seed=7))
    assert np.array_equal(a.gyro, b.gyro)
    assert np.array_equal(a.accel_x, b.accel_x)


def test_noise_free_sensor_matches_physics():
    cfg = IMUConfig(gyro_bias=0.0, gyro_noise_std=0.0, accel_noise_std=0.0,
                    disturbance_amplitude=0.0)
    data = simulate_imu(cfg)
    assert np.allclose(data.gyro, data.true_rate)
    assert np.allclose(data.accel_x, GRAVITY * np.sin(data.true_angle))
    assert np.allclose(data.accel_z, GRAVITY * np.cos(data.true_angle))


def test_gyro_bias_shows_up_in_mean_error():
    cfg = IMUConfig(gyro_bias=np.deg2rad(3.0))
    data = simulate_imu(cfg)
    mean_error = np.mean(data.gyro - data.true_rate)
    assert abs(mean_error - cfg.gyro_bias) < np.deg2rad(0.1)
