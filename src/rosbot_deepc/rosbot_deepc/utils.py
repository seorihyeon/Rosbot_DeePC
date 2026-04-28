import csv
import glob
import math
import os
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

YAW_REPRESENTATION_WRAP = "wrap"
YAW_REPRESENTATION_UNWRAP = "unwrap"
VALID_YAW_REPRESENTATIONS = (
    YAW_REPRESENTATION_WRAP,
    YAW_REPRESENTATION_UNWRAP,
)
TWO_PI = 2.0 * math.pi


@dataclass
class RefPoint:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    w: float


def wrap_to_pi(angle: float) -> float:
    raw = float(angle)
    wrapped = (raw + math.pi) % TWO_PI - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1.0e-12) and raw > 0.0:
        return math.pi
    return wrapped


def signed_angle_diff(angle: float) -> float:
    return wrap_to_pi(angle)


def unwrap_angle(
    new_angle: float,
    prev_angle: Optional[float],
    prev_unwrapped: Optional[float],
) -> float:
    if prev_angle is None or prev_unwrapped is None:
        return new_angle

    delta = signed_angle_diff(new_angle - float(prev_angle))
    return float(prev_unwrapped) + delta


def unwrap_angle_sequence(angles: List[float]) -> List[float]:
    if not angles:
        return []

    out = [float(angles[0])]
    prev_raw = float(angles[0])

    for angle in angles[1:]:
        raw = float(angle)
        delta = signed_angle_diff(raw - prev_raw)
        out.append(out[-1] + delta)
        prev_raw = raw

    return out


def align_reference_yaw_to_reset_branch(points: List[RefPoint]) -> float:
    if not points:
        return 0.0

    reset_yaw = wrap_to_pi(points[0].yaw)
    yaw_offset = reset_yaw - float(points[0].yaw)
    if abs(yaw_offset) <= 1.0e-12:
        return 0.0

    for point in points:
        point.yaw += yaw_offset

    return yaw_offset


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return wrap_to_pi(math.atan2(siny_cosp, cosy_cosp))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def body_frame_pose_error(
    target_x: float,
    target_y: float,
    target_yaw: float,
    x: float,
    y: float,
    yaw: float,
    *,
    wrap_yaw_error: bool = True,
) -> Tuple[float, float, float]:
    dx = float(target_x) - float(x)
    dy = float(target_y) - float(y)
    yaw = float(yaw)
    e_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    e_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    e_psi = float(target_yaw) - yaw
    if wrap_yaw_error:
        e_psi = signed_angle_diff(e_psi)
    return e_x, e_y, e_psi


def unicycle_tracking_law(
    ref_v: float,
    ref_w: float,
    e_x: float,
    e_y: float,
    e_psi: float,
    *,
    kx: float,
    ky: float,
    kpsi: float,
    v_min: float,
    v_max: float,
    w_min: float,
    w_max: float,
) -> Tuple[float, float]:
    cmd_v = float(ref_v) * math.cos(float(e_psi)) + float(kx) * float(e_x)
    cmd_w = float(ref_w) + float(ky) * float(e_y) + float(kpsi) * float(e_psi)
    return clamp(cmd_v, v_min, v_max), clamp(cmd_w, w_min, w_max)


def normalize_yaw_representation(value) -> str:
    rep = str(value).strip().lower()
    if rep in VALID_YAW_REPRESENTATIONS:
        return rep
    raise ValueError(
        "yaw_representation must be one of: wrap, unwrap"
    )


def yaw_representation_uses_unwrapped_scalar(yaw_representation: str) -> bool:
    return normalize_yaw_representation(yaw_representation) == YAW_REPRESENTATION_UNWRAP


def encode_deepc_output(
    *,
    x: float,
    y: float,
    yaw: float,
    yaw_representation: str,
) -> List[float]:
    rep = normalize_yaw_representation(yaw_representation)
    if rep == YAW_REPRESENTATION_WRAP:
        yaw = wrap_to_pi(float(yaw))
    return [float(x), float(y), float(yaw)]


def make_pose_stamped(
    x: float,
    y: float,
    yaw: float,
    *,
    frame_id: str = "odom",
    stamp=None,
):
    from geometry_msgs.msg import PoseStamped

    ps = PoseStamped()
    if stamp is not None:
        ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = 0.0
    ps.pose.orientation.z = math.sin(float(yaw) / 2.0)
    ps.pose.orientation.w = math.cos(float(yaw) / 2.0)
    return ps


def build_path_msg(points, *, frame_id: str = "odom"):
    from nav_msgs.msg import Path

    msg = Path()
    msg.header.frame_id = frame_id
    for point in points:
        msg.poses.append(
            make_pose_stamped(point.x, point.y, point.yaw, frame_id=frame_id)
        )
    return msg


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


def load_reference_csv(
    csv_path: str,
    dt: float,
    append_final_stop_steps: int = 20,
) -> List[RefPoint]:
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
    ts = [_optional_float(r, "t") for r in rows]
    yaws = [_optional_float(r, "yaw") for r in rows]
    vs = [_optional_float(r, "v") for r in rows]
    ws = [_optional_float(r, "w") for r in rows]

    n = len(rows)

    # yaw interpolation
    for i in range(n):
        if yaws[i] is not None:
            continue

        if n == 1:
            yaws[i] = 0.0
            continue

        if i < n - 1:
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
        else:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]

        if abs(dx) + abs(dy) < 1e-9:
            yaws[i] = yaws[i - 1] if i > 0 else 0.0
        else:
            yaws[i] = math.atan2(dy, dx)

    yaws = unwrap_angle_sequence([float(y) for y in yaws])

    # v calculation
    for i in range(n):
        if vs[i] is not None:
            continue

        if i < n - 1:
            ds = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            vs[i] = ds / max(dt, 1e-9)
        else:
            vs[i] = 0.0

    # w calcuation
    for i in range(n):
        if ws[i] is not None:
            continue

        if i < n - 1:
            ws[i] = (yaws[i + 1] - yaws[i]) / max(dt, 1e-9)
        else:
            ws[i] = 0.0

    traj: List[RefPoint] = []
    for i in range(n):
        t = ts[i] if ts[i] is not None else i * dt
        traj.append(
            RefPoint(
                t=t,
                x=xs[i],
                y=ys[i],
                yaw=float(yaws[i]),
                v=float(vs[i]),
                w=float(ws[i]),
            )
        )

    if append_final_stop_steps > 0:
        last = traj[-1]
        for k in range(append_final_stop_steps):
            traj.append(
                RefPoint(
                    t=last.t + (k + 1) * dt,
                    x=last.x,
                    y=last.y,
                    yaw=last.yaw,
                    v=0.0,
                    w=0.0,
                )
            )

    return traj


def block_hankel(data: np.ndarray, L: int) -> np.ndarray:
    """Build a block Hankel matrix from data shaped as (dim, T)."""
    dim, T = data.shape
    if T < L:
        raise ValueError(f"Not enough data for Hankel. T = {T}, L={L}")

    n_col = T - L + 1
    H = np.zeros((dim * L, n_col), dtype=np.float64)
    for i in range(n_col):
        H[:, i] = data[:, i:i + L].reshape(-1, order="F")
    return H


def resolve_dataset_path(
    dataset_csv: str,
    dataset_dir: str,
    dataset_glob: str,
    dataset_mode: str,
) -> list[str]:
    if dataset_mode == "single":
        if dataset_csv:
            if not os.path.isfile(dataset_csv):
                raise FileNotFoundError(f"Dataset file not found: {dataset_csv}")
            return [dataset_csv]

        pattern = os.path.join(dataset_dir, dataset_glob)
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No {dataset_glob} files found in dataset_dir: {dataset_dir}"
            )
        files.sort(key=os.path.getmtime)
        return [files[-1]]

    if dataset_mode == "mosaic":
        pattern = os.path.join(dataset_dir, dataset_glob)
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No files matched: {pattern}")
        files.sort()
        return files

    raise ValueError(f"Unknown dataset_mode: {dataset_mode}")


def load_dataset_csv(
    csv_path: str,
    drop_initial_rows: int = 0,
    max_rows: int = 0,
    yaw_representation: str = YAW_REPRESENTATION_WRAP,
    y_shift_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if drop_initial_rows > 0:
        rows = rows[drop_initial_rows:]

    if max_rows > 0 and len(rows) > max_rows:
        rows = rows[-max_rows:]

    if len(rows) == 0:
        raise RuntimeError("Dataset CSV is empty after dropping initial rows.")

    y_shift_steps = int(y_shift_steps)
    if y_shift_steps < 0:
        raise ValueError("y_shift_steps must be nonnegative")
    if len(rows) <= y_shift_steps:
        raise RuntimeError(
            f"Dataset CSV has {len(rows)} rows, not enough for y_shift_steps={y_shift_steps}."
        )

    required = ["cmd_v", "cmd_w", "x", "y", "yaw"]
    for key in required:
        if key not in rows[0]:
            raise KeyError(f"CSV column '{key}' is missing.")

    yaw_representation = normalize_yaw_representation(yaw_representation)
    raw_yaws = [float(row["yaw"]) for row in rows]
    if yaw_representation == YAW_REPRESENTATION_UNWRAP:
        yaws = unwrap_angle_sequence(raw_yaws)
    else:
        yaws = [wrap_to_pi(yaw) for yaw in raw_yaws]

    u_list = []
    y_list = []
    aligned_len = len(rows) - y_shift_steps

    for idx in range(aligned_len):
        u_row = rows[idx]
        y_row = rows[idx + y_shift_steps]
        yaw = yaws[idx + y_shift_steps]
        u_list.append([
            float(u_row["cmd_v"]),
            float(u_row["cmd_w"]),
        ])
        y_list.append(
            encode_deepc_output(
                x=float(y_row["x"]),
                y=float(y_row["y"]),
                yaw=float(yaw),
                yaw_representation=yaw_representation,
            )
        )

    u_data = np.asarray(u_list, dtype=np.float64).T   # shape (2, T)
    y_data = np.asarray(y_list, dtype=np.float64).T
    return u_data, y_data


def load_multiple_dataset_csvs(
    csv_paths: list[str],
    drop_initial_rows: int = 0,
    min_rows_per_dataset: int = 1,
    max_rows_per_dataset: int = 0,
    yaw_representation: str = YAW_REPRESENTATION_WRAP,
    y_shift_steps: int = 1,
) -> List[dict]:
    datasets: List[dict] = []

    u_dim_expected: Optional[int] = None
    y_dim_expected: Optional[int] = None

    for path in csv_paths:
        try:
            u_data, y_data = load_dataset_csv(
                path,
                drop_initial_rows=drop_initial_rows,
                max_rows=max_rows_per_dataset,
                yaw_representation=yaw_representation,
                y_shift_steps=y_shift_steps,
            )
        except RuntimeError as exc:
            if (
                "empty after dropping initial rows" in str(exc)
                or "not enough for y_shift_steps" in str(exc)
            ):
                continue
            raise

        T = u_data.shape[1]
        if T < min_rows_per_dataset:
            continue

        if u_dim_expected is None:
            u_dim_expected = u_data.shape[0]
            y_dim_expected = y_data.shape[0]
        else:
            if u_data.shape[0] != u_dim_expected or y_data.shape[0] != y_dim_expected:
                raise RuntimeError(
                    f"Dataset dimension mismatch in {path}: "
                    f"got u_dim={u_data.shape[0]}, y_dim={y_data.shape[0]}, "
                    f"expected u_dim={u_dim_expected}, y_dim={y_dim_expected}"
                )

        datasets.append({"path": path, "u_data": u_data, "y_data": y_data})

    if not datasets:
        raise RuntimeError("No usable datasets found for mosaic Hankel.")

    return datasets


def build_mosaic_input_hankel(
    datasets: List[dict],
    L: int,
) -> Tuple[np.ndarray, List[str]]:
    Hu_list: List[np.ndarray] = []
    used_paths: List[str] = []

    for ds in datasets:
        u_data = ds["u_data"]
        if u_data.shape[1] < L:
            continue
        Hu_list.append(block_hankel(u_data, L))
        used_paths.append(ds["path"])

    if not Hu_list:
        raise RuntimeError(f"No dataset has enough length for L={L}")

    Hu = np.concatenate(Hu_list, axis=1)
    return Hu, used_paths


def build_mosaic_hankel(datasets: list[dict], L: int) -> Tuple[np.ndarray, np.ndarray]:
    Hu_list: List[np.ndarray] = []
    Hy_list: List[np.ndarray] = []

    for ds in datasets:
        u_data = ds["u_data"]
        y_data = ds["y_data"]

        if u_data.shape[1] < L:
            continue

        Hu_i = block_hankel(u_data, L)
        Hy_i = block_hankel(y_data, L)

        Hu_list.append(Hu_i)
        Hy_list.append(Hy_i)

    if not Hu_list:
        raise RuntimeError(f"No dataset has enough length for L={L}")

    Hu = np.concatenate(Hu_list, axis=1)
    Hy = np.concatenate(Hy_list, axis=1)
    return Hu, Hy


def numerical_rank(
    A: np.ndarray,
    tol: Optional[float] = None,
) -> Tuple[int, np.ndarray, float]:
    s = np.linalg.svd(A, compute_uv=False)
    if s.size == 0:
        return 0, s, 0.0

    if tol is None or tol <= 0.0:
        tol = max(A.shape) * np.finfo(A.dtype).eps * s[0]

    rank = int(np.sum(s > tol))
    return rank, s, float(tol)


def check_PE_condition(
    order: int,
    u_data: Optional[np.ndarray] = None,
    mosaic_datasets: Optional[List[dict]] = None,
    tol: Optional[float] = None,
) -> dict:
    """Check whether the input Hankel matrix has full row rank."""
    if mosaic_datasets is not None:
        Hu, used_paths = build_mosaic_input_hankel(mosaic_datasets, order)
        m = mosaic_datasets[0]["u_data"].shape[0]
        expected_rank = m * order
        rank, s, used_tol = numerical_rank(Hu, tol)

        return {
            "mode": "mosaic",
            "ok": (rank == expected_rank),
            "order": order,
            "u_dim": m,
            "rank": rank,
            "expected_rank": expected_rank,
            "n_cols": Hu.shape[1],
            "used_dataset_count": len(used_paths),
            "used_paths": used_paths,
            "sigma_max": float(s[0]) if s.size > 0 else 0.0,
            "sigma_min": float(s[-1]) if s.size > 0 else 0.0,
            "tol": used_tol
        }

    if u_data is None:
        raise ValueError("u_data or mosaic_datasets must be provided.")

    Hu = block_hankel(u_data, order)
    m, T = u_data.shape
    expected_rank = m * order
    rank, s, used_tol = numerical_rank(Hu, tol)

    return {
        "mode": "single",
        "ok": (rank == expected_rank),
        "order": order,
        "u_dim": m,
        "rank": rank,
        "expected_rank": expected_rank,
        "n_cols": Hu.shape[1],
        "T": T,
        "min_length_single": (m + 1) * order - 1,
        "sigma_max": float(s[0]) if s.size > 0 else 0.0,
        "sigma_min": float(s[-1]) if s.size > 0 else 0.0,
        "tol": used_tol
    }


def stack_history_with_zero_padding(
    hist: Deque[np.ndarray],
    dim: int,
    Tini: int,
) -> np.ndarray:
    seq: List[np.ndarray] = []

    pad_len = max(0, Tini - len(hist))
    for _ in range(pad_len):
        seq.append(np.zeros(dim, dtype=np.float64))

    for v in hist:
        seq.append(np.asarray(v, dtype=np.float64).reshape(dim))

    return np.concatenate(seq, axis=0)
