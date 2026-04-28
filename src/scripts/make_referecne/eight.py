#!/usr/bin/env python3
import csv
import math
import os

DT = 0.03          # deepc_params.yaml 의 sample_time 과 맞춤
RADIUS = 1.0       # [m]
SPEED = 0.30       # [m/s]
OUT_PATH = "/ws/ref/eight_reference.csv"


def wrap_to_pi(angle: float) -> float:
    raw = float(angle)
    wrapped = (raw + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1.0e-12) and raw > 0.0:
        return math.pi
    return wrapped


def append_circle_segment(
    rows: list[dict],
    *,
    center_x: float,
    center_y: float,
    radius: float,
    speed: float,
    dt: float,
    clockwise: bool,
    start_phase: float,
    start_time: float,
    include_start: bool,
) -> float:
    omega = speed / radius
    signed_omega = -omega if clockwise else omega
    total_time = 2.0 * math.pi * radius / speed
    steps = int(round(total_time / dt)) + 1

    start_idx = 0 if include_start else 1
    for i in range(start_idx, steps):
        tau = min(i * dt, total_time)
        phase = start_phase + signed_omega * tau

        x = center_x + radius * math.cos(phase)
        y = center_y + radius * math.sin(phase)

        if clockwise:
            yaw = wrap_to_pi(phase - math.pi / 2.0)
        else:
            yaw = wrap_to_pi(phase + math.pi / 2.0)

        rows.append({
            "t": f"{start_time + tau:.6f}",
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "yaw": f"{yaw:.6f}",
            "v": f"{speed:.6f}",
            "w": f"{signed_omega:.6f}",
        })

    return start_time + total_time


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    rows: list[dict] = []

    # 위쪽 원: 중심 (0, +R), 원점에서 시작해서 +x heading으로 출발
    t_end = append_circle_segment(
        rows,
        center_x=0.0,
        center_y=RADIUS,
        radius=RADIUS,
        speed=SPEED,
        dt=DT,
        clockwise=False,
        start_phase=-math.pi / 2.0,
        start_time=0.0,
        include_start=True,
    )

    # 아래쪽 원: 중심 (0, -R), 원점을 지나 다시 +x heading으로 이어짐
    append_circle_segment(
        rows,
        center_x=0.0,
        center_y=-RADIUS,
        radius=RADIUS,
        speed=SPEED,
        dt=DT,
        clockwise=True,
        start_phase=math.pi / 2.0,
        start_time=t_end,
        include_start=False,
    )

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "yaw", "v", "w"])
        writer.writeheader()
        writer.writerows(rows)

    total_time = float(rows[-1]["t"]) if rows else 0.0
    omega = SPEED / RADIUS

    print(f"saved: {OUT_PATH}")
    print(
        "figure-eight reference: "
        f"radius={RADIUS:.3f} m, speed={SPEED:.3f} m/s, "
        f"|omega|={omega:.3f} rad/s, total_time={total_time:.3f} s"
    )


if __name__ == "__main__":
    main()
