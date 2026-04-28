from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/ws/.cache/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module="matplotlib.projections",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/ws/datasets/")

CORE_COLUMNS = {
    "step",
    "sim_time_sec",
    "x",
    "y",
    "yaw",
    "v_meas",
    "w_meas",
    "cmd_v",
    "cmd_w",
}

REFERENCE_COLUMNS = {
    "ref_x",
    "ref_y",
    "ref_yaw",
    "ref_v",
    "ref_w",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual inspection tool for collected Rosbot DeePC CSV logs."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "CSV file(s) or directory. If omitted, the newest CSV in <RESULTS_DIR> is used. "
            "If a filename is given and it does not exist, <RESULTS_DIR>/<name> is tried."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNG next to each CSV instead of only showing the figure.",
    )
    parser.add_argument(
        "--trim-start",
        action="store_true",
        help="Trim the initial inactive samples before plotting.",
    )
    parser.add_argument(
        "--motion-eps",
        type=float,
        default=1e-4,
        help="Threshold used to detect when motion/command starts.",
    )
    return parser.parse_args()


def resolve_paths(inputs: Iterable[str]) -> list[Path]:
    inputs = list(inputs)
    if not inputs:
        csv_files = sorted(RESULTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}")
        return [csv_files[0].resolve()]

    resolved: list[Path] = []
    for item in inputs:
        p = Path(item)
        candidates = [p, RESULTS_DIR / item]
        found = None
        for cand in candidates:
            if cand.is_file():
                found = cand.resolve()
                resolved.append(found)
                break
            if cand.is_dir():
                dir_csvs = sorted(cand.glob("*.csv"))
                if not dir_csvs:
                    raise FileNotFoundError(f"No CSV files found in directory: {cand}")
                for csv_path in dir_csvs:
                    resolved.append(csv_path.resolve())
                found = cand.resolve()
                break
        if found is None:
            raise FileNotFoundError(f"Input not found: {item}")
    if not resolved:
        raise FileNotFoundError("No CSV files resolved from inputs.")
    return resolved


def signed_angle_diff(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def has_reference(df: pd.DataFrame) -> bool:
    return REFERENCE_COLUMNS.issubset(df.columns)


def detect_active_start(df: pd.DataFrame, eps: float) -> int:
    signal = (
        df["cmd_v"].abs()
        + df["cmd_w"].abs()
        + df["v_meas"].abs()
        + df["w_meas"].abs()
    )
    active = np.flatnonzero(signal.to_numpy() > eps)
    if len(active) > 0:
        return int(active[0])

    dx = df["x"].diff().fillna(0.0)
    dy = df["y"].diff().fillna(0.0)
    moved = np.flatnonzero(np.hypot(dx, dy).to_numpy() > eps)
    return int(moved[0]) if len(moved) > 0 else 0


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = sorted(CORE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns: {missing}")


def compute_path_length(df: pd.DataFrame) -> float:
    dx = df["x"].diff().fillna(0.0).to_numpy()
    dy = df["y"].diff().fillna(0.0).to_numpy()
    return float(np.hypot(dx, dy).sum())


def summarize(df: pd.DataFrame, active_start: int) -> dict[str, float | str]:
    t = df["sim_time_sec"].to_numpy()
    dt = np.diff(t)

    summary: dict[str, float | str] = {
        "mode": "reference" if has_reference(df) else "random",
        "samples": len(df),
        "duration_sec": float(t[-1] - t[0]) if len(t) >= 2 else 0.0,
        "dt_mean_ms": float(dt.mean() * 1000.0) if len(dt) else 0.0,
        "dt_std_ms": float(dt.std() * 1000.0) if len(dt) else 0.0,
        "path_length_m": compute_path_length(df),
        "active_start_idx": int(active_start),
        "inactive_samples": int(active_start),
        "cmd_v_abs_mean": float(np.abs(df["cmd_v"]).mean()),
        "cmd_w_abs_mean": float(np.abs(df["cmd_w"]).mean()),
        "v_meas_abs_mean": float(np.abs(df["v_meas"]).mean()),
        "w_meas_abs_mean": float(np.abs(df["w_meas"]).mean()),
    }

    if has_reference(df):
        pos_err = np.hypot(df["x"] - df["ref_x"], df["y"] - df["ref_y"]).to_numpy()
        yaw_err = signed_angle_diff((df["yaw"] - df["ref_yaw"]).to_numpy())
        summary["pos_err_mean_cm"] = float(pos_err.mean() * 100.0)
        summary["pos_err_max_cm"] = float(pos_err.max() * 100.0)
        summary["yaw_err_mean_deg"] = float(np.rad2deg(np.abs(yaw_err)).mean())
        summary["yaw_err_max_deg"] = float(np.rad2deg(np.abs(yaw_err)).max())

    return summary


def make_figure(df_raw: pd.DataFrame, csv_path: Path, trim_start: bool, motion_eps: float) -> plt.Figure:
    validate_columns(df_raw, csv_path)

    ref_mode = has_reference(df_raw)
    active_start = detect_active_start(df_raw, motion_eps)
    df = df_raw.iloc[active_start:].reset_index(drop=True) if trim_start else df_raw.copy()

    if len(df) == 0:
        raise ValueError(f"{csv_path.name}: no rows remain after trimming.")

    t0 = float(df["sim_time_sec"].iloc[0])
    t = df["sim_time_sec"].to_numpy() - t0
    dt = np.diff(df["sim_time_sec"].to_numpy(), prepend=df["sim_time_sec"].iloc[0])

    summary = summarize(df_raw, active_start)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Collection Inspection ({summary['mode']}): {csv_path.name}",
        fontsize=14,
    )

    # (0, 0) XY trajectory
    ax = axes[0, 0]
    if ref_mode:
        ax.plot(df["ref_x"], df["ref_y"], label="reference")
    ax.plot(df["x"], df["y"], label="robot")
    if ref_mode:
        ax.scatter(df["ref_x"].iloc[0], df["ref_y"].iloc[0], marker="o", s=30, label="ref start")
        ax.scatter(df["ref_x"].iloc[-1], df["ref_y"].iloc[-1], marker="s", s=30, label="ref end")
    ax.scatter(df["x"].iloc[0], df["y"].iloc[0], marker="x", s=40, label="robot start")
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], marker="^", s=40, label="robot end")
    ax.set_title("XY trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8)

    # (0, 1) tracking error or yaw
    ax = axes[0, 1]
    if ref_mode:
        pos_err = np.hypot(df["x"] - df["ref_x"], df["y"] - df["ref_y"]).to_numpy()
        yaw_err_deg = np.rad2deg(
            signed_angle_diff((df["yaw"] - df["ref_yaw"]).to_numpy())
        )
        ax.plot(t, pos_err, label="position error")
        ax.set_title("Tracking error")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position error [m]")
        ax.grid(True)
        ax2 = ax.twinx()
        ax2.plot(t, yaw_err_deg, linestyle="--", label="yaw error")
        ax2.set_ylabel("yaw error [deg]")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    else:
        yaw_deg = np.rad2deg(df["yaw"].to_numpy())
        ax.plot(t, yaw_deg, label="yaw")
        ax.set_title("Yaw")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("yaw [deg]")
        ax.grid(True)
        ax.legend(fontsize=8)

    # (0, 2) linear velocity
    ax = axes[0, 2]
    if ref_mode:
        ax.plot(t, df["ref_v"], label="ref_v")
    ax.plot(t, df["v_meas"], label="v_meas")
    ax.plot(t, df["cmd_v"], label="cmd_v")
    if "cmd_v_nom" in df.columns:
        ax.plot(t, df["cmd_v_nom"], linestyle="--", label="cmd_v_nom")
    ax.set_title("Linear velocity")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("v [m/s]")
    ax.grid(True)
    ax.legend(fontsize=8)

    # (1, 0) angular velocity
    ax = axes[1, 0]
    if ref_mode:
        ax.plot(t, df["ref_w"], label="ref_w")
    ax.plot(t, df["w_meas"], label="w_meas")
    ax.plot(t, df["cmd_w"], label="cmd_w")
    if "cmd_w_nom" in df.columns:
        ax.plot(t, df["cmd_w_nom"], linestyle="--", label="cmd_w_nom")
    ax.set_title("Angular velocity")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("w [rad/s]")
    ax.grid(True)
    ax.legend(fontsize=8)

    # (1, 1) perturbation or generic motion/command magnitude
    ax = axes[1, 1]
    has_dv = "dv_pert" in df.columns
    has_dw = "dw_pert" in df.columns
    if has_dv or has_dw:
        if has_dv:
            ax.plot(t, df["dv_pert"], label="dv_pert")
        if has_dw:
            ax.plot(t, df["dw_pert"], label="dw_pert")
        ax.set_title("Injected perturbation")
        ax.set_ylabel("perturbation")
        ax.legend(fontsize=8)
    else:
        cmd_norm = np.sqrt(df["cmd_v"].to_numpy() ** 2 + 0.1 * df["cmd_w"].to_numpy() ** 2)
        meas_norm = np.sqrt(df["v_meas"].to_numpy() ** 2 + 0.1 * df["w_meas"].to_numpy() ** 2)
        ax.plot(t, cmd_norm, label="command magnitude")
        ax.plot(t, meas_norm, label="measured magnitude")
        ax.set_title("Command / motion magnitude")
        ax.set_ylabel("magnitude")
        ax.legend(fontsize=8)
    ax.set_xlabel("time [s]")
    ax.grid(True)

    # (1, 2) sample time
    ax = axes[1, 2]
    ax.plot(t, dt * 1000.0, label="dt")
    ax.axhline(np.mean(dt) * 1000.0, linestyle="--", label="mean dt")
    ax.set_title("Sample time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("dt [ms]")
    ax.grid(True)
    ax.legend(fontsize=8)

    summary_lines = [
        f"mode          : {summary['mode']}",
        f"reference     : {df_raw['reference_name'].iloc[0] if 'reference_name' in df_raw.columns else '-'}",
        f"samples       : {summary['samples']}",
        f"duration      : {summary['duration_sec']:.2f} s",
        f"mean dt       : {summary['dt_mean_ms']:.2f} ms",
        f"std dt        : {summary['dt_std_ms']:.2f} ms",
        f"path length    : {summary['path_length_m']:.2f} m",
        f"|cmd_v| mean   : {summary['cmd_v_abs_mean']:.3f}",
        f"|cmd_w| mean   : {summary['cmd_w_abs_mean']:.3f}",
        f"|v_meas| mean  : {summary['v_meas_abs_mean']:.3f}",
        f"|w_meas| mean  : {summary['w_meas_abs_mean']:.3f}",
    ]

    if ref_mode:
        summary_lines.extend([
            f"mean pos err  : {summary['pos_err_mean_cm']:.2f} cm",
            f"max pos err   : {summary['pos_err_max_cm']:.2f} cm",
            f"mean yaw err  : {summary['yaw_err_mean_deg']:.2f} deg",
            f"max yaw err   : {summary['yaw_err_max_deg']:.2f} deg",
        ])

    summary_lines.extend([
        f"inactive head : {summary['inactive_samples']} samples",
        f"trimmed start : {'yes' if trim_start else 'no'}",
    ])

    fig.text(
        0.73,
        0.08,
        "\n".join(summary_lines),
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    if not trim_start and active_start > 0:
        inactive_end_t = float(df_raw["sim_time_sec"].iloc[active_start] - df_raw["sim_time_sec"].iloc[0])
        for axis in [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
            axis.axvspan(0.0, inactive_end_t, alpha=0.12)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    args = parse_args()
    csv_paths = resolve_paths(args.inputs)

    figures = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        fig = make_figure(df, csv_path, trim_start=args.trim_start, motion_eps=args.motion_eps)
        figures.append(fig)

        if args.save:
            out_path = csv_path.with_suffix(".inspect.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
