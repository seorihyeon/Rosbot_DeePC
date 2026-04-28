#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path('/ws/results')

PRED_COMMON_REQUIRED = {
    'step', 'mode', 'sim_time_sec', 'ref_idx', 'pred_step',
    'u_v', 'u_w', 'y_x', 'y_y'
}
PRED_YAW_SCALAR_REQUIRED = {'y_yaw'}

RUN_REQUIRED = {
    'step', 'mode', 'sim_time_sec', 'ref_idx',
    'ref_x', 'ref_y', 'ref_yaw', 'ref_v', 'ref_w',
    'x', 'y', 'yaw', 'v_meas', 'w_meas',
    'e_x', 'e_y', 'e_psi', 'cmd_v', 'cmd_w'
}


def unwrap_angle_series(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    return np.unwrap(arr)


def resolve_prediction_csv_path(arg: Optional[str]) -> Path:
    if arg:
        path = Path(arg)
        if path.exists():
            return path.resolve()
        candidate = RESULTS_DIR / arg
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Prediction CSV file not found: '{arg}'\n"
            f"- not found in both {path} and {candidate}"
        )

    csv_files = sorted(
        RESULTS_DIR.glob('*prediction*.csv'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No prediction CSV found in {RESULTS_DIR}.\n"
            "Pass the prediction CSV path explicitly."
        )
    return csv_files[0].resolve()


def infer_run_csv_path(pred_path: Path) -> Optional[Path]:
    name = pred_path.name
    candidates = []

    if '_prediction_' in name:
        candidates.append(pred_path.with_name(name.replace('_prediction_', '_', 1)))

    if name.startswith('deepc_prediction_'):
        candidates.append(pred_path.with_name(name.replace('deepc_prediction_', 'deepc_run_', 1)))

    if name.startswith('prediction_'):
        candidates.append(pred_path.with_name(name.replace('prediction_', 'run_', 1)))

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    if pred_path.parent.exists():
        stem_tokens = name.replace('.csv', '').split('_')
        if len(stem_tokens) >= 2:
            tail = '_'.join(stem_tokens[-2:])
            nearby = sorted(pred_path.parent.glob(f'*{tail}.csv'))
            nearby = [p for p in nearby if 'prediction' not in p.name]
            if nearby:
                return nearby[-1].resolve()

    return None


def load_prediction_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_common = PRED_COMMON_REQUIRED - set(df.columns)
    if missing_common:
        raise ValueError(f'Prediction CSV missing columns: {sorted(missing_common)}')

    df = df.copy().sort_values(['step', 'pred_step']).reset_index(drop=True)

    cols = set(df.columns)
    if PRED_YAW_SCALAR_REQUIRED.issubset(cols):
        df['yaw_pred'] = df['y_yaw'].astype(np.float64)
        df['yaw_repr'] = 'scalar_yaw'
    else:
        raise ValueError("Prediction CSV must contain ['y_yaw'].")

    return df


def load_run_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = RUN_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'Run CSV missing columns: {sorted(missing)}')

    df = df.copy().sort_values('step').reset_index(drop=True)
    df['yaw_unwrapped'] = unwrap_angle_series(df['yaw'].to_numpy(dtype=np.float64))
    df['ref_yaw_unwrapped'] = unwrap_angle_series(df['ref_yaw'].to_numpy(dtype=np.float64))
    return df


def _align_prediction_branch_per_step(err_df: pd.DataFrame) -> np.ndarray:
    """
    Align each predicted horizon to the closest 2*pi branch of the actual yaw
    at that horizon's first aligned sample.

    This keeps cumulative yaw error meaningful while remaining robust to
    scalar-yaw prediction logs.
    """
    out = np.zeros(len(err_df), dtype=np.float64)

    tmp = err_df[['step', 'pred_step', 'yaw_pred', 'yaw_actual']].copy()
    tmp = tmp.sort_values(['step', 'pred_step'])

    two_pi = 2.0 * np.pi

    for _, seg in tmp.groupby('step', sort=False):
        pred0 = float(seg['yaw_pred'].iloc[0])
        actual0 = float(seg['yaw_actual'].iloc[0])

        k = np.round((actual0 - pred0) / two_pi)
        shift = two_pi * k

        out[seg.index.to_numpy()] = seg['yaw_pred'].to_numpy(dtype=np.float64) + shift

    return out


def build_error_df(pred_df: pd.DataFrame, run_df: pd.DataFrame, actual_offset: int) -> pd.DataFrame:
    actual_df = run_df[['step', 'x', 'y', 'yaw_unwrapped']].copy()
    actual_df = actual_df.rename(
        columns={
            'step': 'actual_step',
            'x': 'x_actual',
            'y': 'y_actual',
            'yaw_unwrapped': 'yaw_actual',
        }
    )

    err_df = pred_df.copy()
    err_df['actual_step'] = err_df['step'] + err_df['pred_step'] + actual_offset
    err_df = err_df.merge(actual_df, on='actual_step', how='inner')
    err_df = err_df.sort_values(['step', 'pred_step']).reset_index(drop=True)

    dx = err_df['y_x'] - err_df['x_actual']
    dy = err_df['y_y'] - err_df['y_actual']

    err_df['yaw_pred_aligned'] = _align_prediction_branch_per_step(err_df)
    err_df['yaw_err_signed'] = err_df['yaw_pred_aligned'] - err_df['yaw_actual']
    err_df['yaw_err'] = np.abs(err_df['yaw_err_signed'])

    err_df['pos_err'] = np.hypot(dx, dy)
    err_df['x_err'] = dx
    err_df['y_err'] = dy
    return err_df


def plot_prediction_fan(ax, pred_df: pd.DataFrame, run_df: Optional[pd.DataFrame], stride: int, max_lines: int) -> None:
    if run_df is not None:
        ax.plot(run_df['ref_x'], run_df['ref_y'], label='reference')
        ax.plot(run_df['x'], run_df['y'], label='actual')

    sampled_steps = pred_df['step'].drop_duplicates().iloc[::max(1, stride)]
    if max_lines > 0:
        sampled_steps = sampled_steps.iloc[:max_lines]

    first = True
    for step in sampled_steps:
        seg = pred_df[pred_df['step'] == step]
        ax.plot(
            seg['y_x'], seg['y_y'],
            alpha=0.35,
            linewidth=1.0,
            label='predicted horizon' if first else None,
        )
        first = False

    ax.set_title('Predicted horizon fan on XY plane')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()


def plot_one_step_prediction(ax, err_df: pd.DataFrame) -> None:
    one = err_df[err_df['pred_step'] == err_df['pred_step'].min()].copy()
    if one.empty:
        ax.set_title('No aligned one-step prediction data')
        return

    ax.plot(one['step'], one['pos_err'], label='one-step position error [m]')
    ax.plot(one['step'], one['yaw_err'], label='one-step yaw error [rad]')
    ax.set_title('One-step prediction error over time')
    ax.set_xlabel('controller step')
    ax.grid(True)
    ax.legend()


def plot_rmse_vs_horizon(ax, err_df: pd.DataFrame) -> None:
    if err_df.empty:
        ax.set_title('No aligned prediction/actual data')
        return

    grouped = err_df.groupby('pred_step').agg(
        pos_rmse=('pos_err', lambda s: float(np.sqrt(np.mean(np.square(s))))),
        yaw_rmse=('yaw_err', lambda s: float(np.sqrt(np.mean(np.square(s))))),
        count=('pos_err', 'size'),
    ).reset_index()

    ax.plot(grouped['pred_step'], grouped['pos_rmse'], marker='o', label='position RMSE [m]')
    ax.plot(grouped['pred_step'], grouped['yaw_rmse'], marker='o', label='yaw RMSE [rad]')
    ax.set_title('Prediction error vs horizon index')
    ax.set_xlabel('pred_step')
    ax.grid(True)
    ax.legend()


def plot_position_error_heatmap(ax, err_df: pd.DataFrame) -> None:
    if err_df.empty:
        ax.set_title('No aligned prediction/actual data')
        return

    pivot = err_df.pivot_table(index='step', columns='pred_step', values='pos_err', aggfunc='mean')
    im = ax.imshow(pivot.to_numpy(), aspect='auto', origin='lower')
    ax.set_title('Position error heatmap')
    ax.set_xlabel('pred_step')
    ax.set_ylabel('controller step')
    plt.colorbar(im, ax=ax, label='position error [m]')


def plot_input_sanity(ax1, ax2, pred_df: pd.DataFrame, v_limit: Optional[float], w_limit: Optional[float]) -> None:
    uv = pred_df.pivot_table(index='step', columns='pred_step', values='u_v', aggfunc='mean')
    uw = pred_df.pivot_table(index='step', columns='pred_step', values='u_w', aggfunc='mean')

    im1 = ax1.imshow(uv.to_numpy(), aspect='auto', origin='lower')
    ax1.set_title('Predicted linear velocity (u_v)')
    ax1.set_xlabel('pred_step')
    ax1.set_ylabel('controller step')
    plt.colorbar(im1, ax=ax1, label='u_v [m/s]')

    im2 = ax2.imshow(uw.to_numpy(), aspect='auto', origin='lower')
    ax2.set_title('Predicted angular velocity (u_w)')
    ax2.set_xlabel('pred_step')
    ax2.set_ylabel('controller step')
    plt.colorbar(im2, ax=ax2, label='u_w [rad/s]')

    if v_limit is not None:
        uv_abs_max = float(np.nanmax(np.abs(uv.to_numpy())))
        ax1.text(
            0.01, 0.98,
            f'|u_v| max = {uv_abs_max:.3f}\nlimit = {v_limit:.3f}',
            transform=ax1.transAxes,
            va='top',
        )
    if w_limit is not None:
        uw_abs_max = float(np.nanmax(np.abs(uw.to_numpy())))
        ax2.text(
            0.01, 0.98,
            f'|u_w| max = {uw_abs_max:.3f}\nlimit = {w_limit:.3f}',
            transform=ax2.transAxes,
            va='top',
        )


def print_summary(pred_df: pd.DataFrame, err_df: Optional[pd.DataFrame], v_limit: Optional[float], w_limit: Optional[float]) -> None:
    print('\n=== Prediction CSV summary ===')
    print(f"controller steps : {pred_df['step'].nunique()}")
    print(f"horizon length   : {pred_df['pred_step'].nunique()}")
    print(f"yaw repr         : {pred_df['yaw_repr'].iloc[0]}")
    print(f"u_v range        : {pred_df['u_v'].min():+.4f} ~ {pred_df['u_v'].max():+.4f}")
    print(f"u_w range        : {pred_df['u_w'].min():+.4f} ~ {pred_df['u_w'].max():+.4f}")
    print(f"yaw_pred range   : {pred_df['yaw_pred'].min():+.4f} ~ {pred_df['yaw_pred'].max():+.4f}")

    if v_limit is not None:
        bad_v = int((pred_df['u_v'].abs() > v_limit + 1e-9).sum())
        print(f"u_v limit violation count (> {v_limit:.3f}) : {bad_v}")
    if w_limit is not None:
        bad_w = int((pred_df['u_w'].abs() > w_limit + 1e-9).sum())
        print(f"u_w limit violation count (> {w_limit:.3f}) : {bad_w}")

    if err_df is not None and not err_df.empty:
        one = err_df[err_df['pred_step'] == err_df['pred_step'].min()]
        print('\n=== Prediction accuracy summary ===')
        print(f"aligned pairs              : {len(err_df)}")
        print(f"one-step pos RMSE [m]      : {np.sqrt(np.mean(one['pos_err'] ** 2)):.4f}")
        print(f"one-step yaw RMSE [rad]    : {np.sqrt(np.mean(one['yaw_err'] ** 2)):.4f}")
        print(f"all-horizon pos RMSE [m]   : {np.sqrt(np.mean(err_df['pos_err'] ** 2)):.4f}")
        print(f"all-horizon yaw RMSE [rad] : {np.sqrt(np.mean(err_df['yaw_err'] ** 2)):.4f}")
    elif err_df is not None:
        print('\nNo aligned actual trajectory samples were found. Try --actual-offset 0 or 1.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize DeePC prediction quality from prediction CSV and optional run CSV.'
    )
    parser.add_argument(
        'prediction_csv',
        nargs='?',
        default=None,
        help='Prediction CSV path. If omitted, use the most recent *prediction*.csv in /ws/results.',
    )
    parser.add_argument(
        '--run-csv',
        default=None,
        help='Run CSV path for actual-vs-prediction comparison. If omitted, try to infer from filename.',
    )
    parser.add_argument(
        '--actual-offset',
        type=int,
        default=1,
        help='Actual alignment offset. Try 1 first; if mismatch looks shifted, try 0.',
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Draw one predicted horizon every N controller steps on XY plot.',
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=80,
        help='Maximum number of predicted horizon lines to draw on XY plot.',
    )
    parser.add_argument(
        '--v-limit',
        type=float,
        default=None,
        help='Optional linear velocity limit for sanity check text.',
    )
    parser.add_argument(
        '--w-limit',
        type=float,
        default=None,
        help='Optional angular velocity limit for sanity check text.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_path = resolve_prediction_csv_path(args.prediction_csv)
    run_path = None

    if args.run_csv:
        candidate = Path(args.run_csv)
        if candidate.exists():
            run_path = candidate.resolve()
        else:
            candidate2 = RESULTS_DIR / args.run_csv
            if candidate2.exists():
                run_path = candidate2.resolve()
            else:
                raise FileNotFoundError(f'Run CSV not found: {args.run_csv}')
    else:
        run_path = infer_run_csv_path(pred_path)

    print(f'Using prediction CSV: {pred_path}')
    if run_path is not None:
        print(f'Using run CSV       : {run_path}')
    else:
        print('Using run CSV       : None (prediction-only mode)')

    pred_df = load_prediction_df(pred_path)
    run_df = load_run_df(run_path) if run_path is not None else None
    err_df = build_error_df(pred_df, run_df, args.actual_offset) if run_df is not None else None

    print_summary(pred_df, err_df, args.v_limit, args.w_limit)

    if run_df is not None:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_one = fig.add_subplot(gs[0, 1])
        ax_rmse = fig.add_subplot(gs[0, 2])
        ax_heat = fig.add_subplot(gs[1, 0])
        ax_uv = fig.add_subplot(gs[1, 1])
        ax_uw = fig.add_subplot(gs[1, 2])

        plot_prediction_fan(ax_xy, pred_df, run_df, args.stride, args.max_lines)
        plot_one_step_prediction(ax_one, err_df)
        plot_rmse_vs_horizon(ax_rmse, err_df)
        plot_position_error_heatmap(ax_heat, err_df)
        plot_input_sanity(ax_uv, ax_uw, pred_df, args.v_limit, args.w_limit)

        fig.suptitle(f'DeePC prediction check\n{pred_path.name}', fontsize=14)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2)
        ax_xy = fig.add_subplot(gs[:, 0])
        ax_uv = fig.add_subplot(gs[0, 1])
        ax_uw = fig.add_subplot(gs[1, 1])

        plot_prediction_fan(ax_xy, pred_df, None, args.stride, args.max_lines)
        plot_input_sanity(ax_uv, ax_uw, pred_df, args.v_limit, args.w_limit)

        fig.suptitle(f'DeePC prediction fan / sanity check\n{pred_path.name}', fontsize=14)
        fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
