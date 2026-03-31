import math
import csv
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RefPoint:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    w: float

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _optional_float(row: dict, key: str) -> Optional[float]:
    if key not in row:
        return None
    value = row[key]
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)

def load_reference_csv(csv_path: str, dt: float, append_final_stop_steps: int = 20) -> List[RefPoint]:
    if not csv_path:
        raise ValueError("reference_csv parameter is empty.")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Reference CSV file not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Reference csv is empty: {csv_path}")
    
    for key in ("x", "y"):
        if key not in rows[0]:
            raise KeyError(f"Reference csv must contain '{key}' column.")

    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    ts = [_optional_float(r,"t") for r in rows]
    yaws = [_optional_float(r,"yaw") for r in rows]
    vs = [_optional_float(r,"v") for r in rows]
    ws = [_optional_float(r,"w") for r in rows]

    n = len(rows)

    # yaw interpolation
    for i in range(n):
        if yaws[i] is not None:
            continue

        if n == 1:
            yaws[i] = 0.0
            continue
        
        if i < n-1:
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
        else:
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
        
        if abs(dx) + abs(dy) < 1e-9:
            yaws[i] = yaws[i-1] if i > 0 else 0.0
        else:
            yaws[i] = math.atan2(dy, dx)

    # v calculation
    for i in range(n):
        if vs[i] is not None:
            continue

        if i < n-1:
            ds = math.hypot(xs[i+1] - xs[i], ys[i+1], ys[i])
            vs[i] = ds/max(dt,1e-9)
        else:
            vs[i] = 0.0

    # w calcuation
    for i in range(n):
        if ws[i] is not None:
            continue
        
        if i < n-1:
            ws[i] = wrap_to_pi(yaws[i+1] - yaws[i])/max(dt,1e-9)
        else:
            ws[i] = 0.0
        
    traj: List[RefPoint] = []
    for i in range(n):
        t = ts[i] if ts[i] is not None else i*dt
        traj.append(RefPoint(t=t, x=xs[i], y=ys[i], yaw=float(yaws[i]), v=float(vs[i]), w=float(ws[i])))

    if append_final_stop_steps > 0:
        last = traj[-1]
        for k in range(append_final_stop_steps):
            traj.append(RefPoint(t=last.t + (k+1)*dt, x=last.x, y=last.y, yaw=last.yaw, v=0.0, w=0.0))

    return traj
