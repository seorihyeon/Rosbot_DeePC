import csv
from pathlib import Path

import numpy as np


def cubic_bezier(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n, endpoint=True)
    omt = 1.0 - t
    pts = (
        (omt**3)[:, None] * p0
        + (3.0 * omt**2 * t)[:, None] * p1
        + (3.0 * omt * t**2)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    return pts


def build_hourglass_curve():
    c = np.array([0.0, 0.0])        # center crossing
    ul = np.array([-0.95, 0.88])    # upper-left
    ur = np.array([0.88, 0.82])     # upper-right
    ll = np.array([-0.95, -0.88])   # lower-left
    lr = np.array([0.95, -0.88])    # lower-right

    segments = [
        # center -> upper-left
        cubic_bezier(
            c,
            np.array([-0.35, 0.20]),
            np.array([-1.05, 0.45]),
            ul,
            90,
        ),
        # upper-left -> upper-right (top bulge)
        cubic_bezier(
            ul,
            np.array([-0.30, 1.30]),
            np.array([0.45, 1.10]),
            ur,
            120,
        ),
        # upper-right -> center
        cubic_bezier(
            ur,
            np.array([1.05, 0.55]),
            np.array([0.35, 0.20]),
            c,
            90,
        ),
        # center -> lower-left
        cubic_bezier(
            c,
            np.array([-0.35, -0.20]),
            np.array([-1.05, -0.55]),
            ll,
            90,
        ),
        # lower-left -> lower-right (bottom bulge)
        cubic_bezier(
            ll,
            np.array([-0.45, -1.20]),
            np.array([0.45, -1.20]),
            lr,
            120,
        ),
        # lower-right -> center
        cubic_bezier(
            lr,
            np.array([1.05, -0.55]),
            np.array([0.35, -0.20]),
            c,
            90,
        ),
    ]

    points = [segments[0]]
    for seg in segments[1:]:
        points.append(seg[1:])  # eliminate duplicated point
    return np.vstack(points)


def resample_by_arclength(points, ds):
    diffs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])

    total_len = s[-1]
    s_new = np.arange(0.0, total_len, ds)
    if s_new[-1] < total_len:
        s_new = np.append(s_new, total_len)

    x_new = np.interp(s_new, s, points[:, 0])
    y_new = np.interp(s_new, s, points[:, 1])
    return np.column_stack([x_new, y_new])


def wrap_to_pi(angle):
    raw = float(angle)
    wrapped = (raw + np.pi) % (2.0 * np.pi) - np.pi
    if np.isclose(wrapped, -np.pi, atol=1.0e-12) and raw > 0.0:
        return np.pi
    return wrapped


def compute_reference_columns(points, dt):
    x = points[:, 0]
    y = points[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.unwrap(np.arctan2(dy, dx))

    step_dx = np.diff(x, append=x[-1])
    step_dy = np.diff(y, append=y[-1])
    v = np.hypot(step_dx, step_dy) / dt

    yaw_wrapped = np.array([wrap_to_pi(a) for a in yaw])
    yaw_unwrapped = np.unwrap(yaw_wrapped)
    w = np.gradient(yaw_unwrapped, dt)

    v[-1] = 0.0
    w[-1] = 0.0

    t = np.arange(len(points)) * dt
    return t, x, y, yaw_wrapped, v, w


def append_final_stop(t, x, y, yaw, v, w, dt, final_stop_steps):
    if final_stop_steps <= 0:
        return t, x, y, yaw, v, w

    t_last = t[-1]
    x_last = x[-1]
    y_last = y[-1]
    yaw_last = yaw[-1]

    t_extra = t_last + dt * np.arange(1, final_stop_steps + 1)
    x_extra = np.full(final_stop_steps, x_last)
    y_extra = np.full(final_stop_steps, y_last)
    yaw_extra = np.full(final_stop_steps, yaw_last)
    v_extra = np.zeros(final_stop_steps)
    w_extra = np.zeros(final_stop_steps)

    return (
        np.concatenate([t, t_extra]),
        np.concatenate([x, x_extra]),
        np.concatenate([y, y_extra]),
        np.concatenate([yaw, yaw_extra]),
        np.concatenate([v, v_extra]),
        np.concatenate([w, w_extra]),
    )


def save_reference_csv(csv_path, t, x, y, yaw, v, w):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "yaw", "v", "w"])
        for row in zip(t, x, y, yaw, v, w):
            writer.writerow([f"{val:.6f}" for val in row])

    print(f"saved: {csv_path}")


def preview(points):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available. preview is omitted.")
        return

    plt.figure(figsize=(6, 8))
    plt.plot(points[:, 0], points[:, 1])
    plt.scatter(points[0, 0], points[0, 1], s=40, label="start")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Hourglass reference")
    plt.show()


def main():
    dt = 0.03
    ref_speed = 0.25
    final_stop_steps = 20
    csv_path = "/ws/ref/hourglass_reference.csv"

    raw_curve = build_hourglass_curve()

    ds = ref_speed * dt
    sampled = resample_by_arclength(raw_curve, ds)

    t, x, y, yaw, v, w = compute_reference_columns(sampled, dt)

    t, x, y, yaw, v, w = append_final_stop(
        t, x, y, yaw, v, w, dt, final_stop_steps
    )

    save_reference_csv(csv_path, t, x, y, yaw, v, w)
    
    preview(np.column_stack([x, y]))


if __name__ == "__main__":
    main()
