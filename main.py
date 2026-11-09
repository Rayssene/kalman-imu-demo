"""Run the full demo: simulate an IMU, run both filters, print errors, save the plot.

Usage:
    python main.py [--alpha 0.98] [--seed 42] [--output docs/angle_comparison.png]
"""

from __future__ import annotations

import argparse

import numpy as np

from kalman_imu.filters import accel_angle, rmse, run_complementary, run_kalman
from kalman_imu.plotting import plot_comparison
from kalman_imu.simulation import IMUConfig, simulate_imu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=0.98,
                        help="complementary filter coefficient (default 0.98)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--output", default="docs/angle_comparison.png",
                        help="where to save the figure")
    args = parser.parse_args()

    # 1. Simulate the sensor.
    cfg = IMUConfig(seed=args.seed)
    data = simulate_imu(cfg)

    # 2. Estimate the angle three ways.
    acc = accel_angle(data.accel_x, data.accel_z)
    comp = run_complementary(data.gyro, acc, cfg.dt, alpha=args.alpha)
    kal, bias = run_kalman(data.gyro, acc, cfg.dt)

    # 3. Report.
    print(f"Simulated {len(data.t)} samples at {1 / cfg.dt:.0f} Hz\n")
    print(f"{'Estimator':<22}{'RMSE [deg]':>12}")
    for name, est in [("Accelerometer only", acc),
                      ("Complementary filter", comp),
                      ("Kalman filter", kal)]:
        print(f"{name:<22}{np.rad2deg(rmse(est, data.true_angle)):>12.2f}")
    print(f"\nGyro bias: true {np.rad2deg(cfg.gyro_bias):.2f} deg/s, "
          f"Kalman estimate {np.rad2deg(bias[-1]):.2f} deg/s")

    # 4. Plot.
    path = plot_comparison(data.t, data.true_angle, acc, comp, kal, args.output)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    main()
