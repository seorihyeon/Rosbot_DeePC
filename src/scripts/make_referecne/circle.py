#!/usr/bin/env python3
import csv
import math
import os

DT = 0.03          # deepc_params.yaml 의 sample_time 과 맞춤
RADIUS = 1.0       # [m]
SPEED = 0.30       # [m/s]
CENTER_X = 0.0
CENTER_Y = RADIUS  # 시작점을 (0, 0), 시작 yaw를 0 rad로 맞추기 위한 설정
OUT_PATH = "/ws/ref/circle_reference.csv"

def wrap_to_pi(angle: float) -> float:
    raw = float(angle)
    wrapped = (raw + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1.0e-12) and raw > 0.0:
        return math.pi
    return wrapped

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    omega = SPEED / RADIUS              # 원운동 각속도
    total_time = 2.0 * math.pi * RADIUS / SPEED
    steps = int(round(total_time / DT)) + 1

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "yaw", "v", "w"])
        writer.writeheader()

        for i in range(steps):
            t = min(i * DT, total_time)

            # 시작점: (0, -RADIUS), 시작 heading: +x 방향
            theta = -math.pi / 2.0 + omega * t

            x = CENTER_X + RADIUS * math.cos(theta)
            y = CENTER_Y + RADIUS * math.sin(theta)
            yaw = wrap_to_pi(theta + math.pi / 2.0)

            writer.writerow({
                "t": f"{t:.6f}",
                "x": f"{x:.6f}",
                "y": f"{y:.6f}",
                "yaw": f"{yaw:.6f}",
                "v": f"{SPEED:.6f}",
                "w": f"{omega:.6f}",
            })

    print(f"saved: {OUT_PATH}")
    print(f"radius={RADIUS:.3f} m, speed={SPEED:.3f} m/s, omega={omega:.3f} rad/s, total_time={total_time:.3f} s")

if __name__ == "__main__":
    main()
