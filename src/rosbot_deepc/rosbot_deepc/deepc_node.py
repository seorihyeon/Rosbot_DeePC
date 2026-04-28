import csv
import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

import rclpy

from .deepc_solver import DeePCSolver
from .tracking_base import TrackingBase
from .utils import (
    RefPoint,
    check_PE_condition,
    clamp,
    encode_deepc_output,
    load_dataset_csv,
    load_multiple_dataset_csvs,
    resolve_dataset_path,
    stack_history_with_zero_padding,
    unicycle_tracking_law,
    wrap_to_pi,
)


class DeePCNode(TrackingBase):
    def __init__(self) -> None:
        super().__init__("deepc_node")

        self._declare_deepc_parameters()
        self._load_deepc_parameters()
        self._init_deepc_state()

        self._load_reference()
        self._load_io_data()
        self._check_PE_condition()

        self._build_solver()
        self.schedule_startup(self.begin_tracking)

        self.get_logger().info("DeePC tracking node started.")
        self.get_logger().info(f"reference_csv : {self.reference_csv}")
        self.get_logger().info(f"dataset_mode  : {self.dataset_mode}")
        self.get_logger().info(f"dataset_info  : {self.dataset_desc}")
        self.get_logger().info(f"yaw_representation : {self.yaw_representation}")
        self.get_logger().info(f"dataset_y_shift_steps : {self.dataset_y_shift_steps}")
        self.get_logger().info(f"reset_before_start : {self.reset_before_start}")
        self.get_logger().info(f"Tini = {self.Tini}, N = {self.N}")
        self.get_logger().info(f"cmd_topic    : {self.cmd_topic}")
        self.get_logger().info(f"odom_topic   : {self.odom_topic}")
        self.get_logger().info(
            f"warmup       : {'enabled' if self.enable_warmup else 'disabled'} "
            f"({self.warmup_steps} steps)"
        )

    def _declare_deepc_parameters(self) -> None:
        self.declare_parameter("kx", 0.8)
        self.declare_parameter("ky", 1.8)
        self.declare_parameter("kpsi", 2.0)

        self.declare_parameter("dataset_mode", "single")
        self.declare_parameter("dataset_csv", "")
        self.declare_parameter("dataset_dir", "/ws/datasets")
        self.declare_parameter("dataset_glob", "*.csv")
        self.declare_parameter("dataset_y_shift_steps", 1)
        self.declare_parameter("drop_initial_rows", 20)
        self.declare_parameter("min_rows_per_dataset", 1)
        self.declare_parameter("max_rows_per_dataset", 0)

        self.declare_parameter("check_pe_before_start", True)
        self.declare_parameter("pe_order_extra", 3)
        self.declare_parameter("pe_rank_tol", 1.0e-9)
        self.declare_parameter("abort_on_pe_failure", True)

        self.declare_parameter("Tini", 12)
        self.declare_parameter("horizon", 15)
        self.declare_parameter("Q_diag", [30.0, 45.0, 12.0])
        self.declare_parameter("R_diag", [1.0, 0.2])
        self.declare_parameter("lambda_g", 1.0)
        self.declare_parameter("lambda_s", 1.0e6)

        self.declare_parameter("solver_name", "OSQP")
        self.declare_parameter("enable_warmup", True)

    def _load_deepc_parameters(self) -> None:
        self.kx = float(self.get_parameter("kx").value)
        self.ky = float(self.get_parameter("ky").value)
        self.kpsi = float(self.get_parameter("kpsi").value)

        self.dataset_mode = str(self.get_parameter("dataset_mode").value)
        self.dataset_csv = str(self.get_parameter("dataset_csv").value)
        self.dataset_dir = str(self.get_parameter("dataset_dir").value)
        self.dataset_glob = str(self.get_parameter("dataset_glob").value)
        self.dataset_y_shift_steps = int(
            self.get_parameter("dataset_y_shift_steps").value
        )
        if self.dataset_y_shift_steps not in (0, 1):
            raise ValueError("dataset_y_shift_steps must be 0 or 1")
        self.drop_initial_rows = int(self.get_parameter("drop_initial_rows").value)
        self.min_rows_per_dataset = int(
            self.get_parameter("min_rows_per_dataset").value
        )
        self.max_rows_per_dataset = int(
            self.get_parameter("max_rows_per_dataset").value
        )

        self.check_pe_before_start = bool(
            self.get_parameter("check_pe_before_start").value
        )
        self.pe_order_extra = int(self.get_parameter("pe_order_extra").value)
        self.pe_rank_tol = float(self.get_parameter("pe_rank_tol").value)
        self.abort_on_pe_failure = bool(
            self.get_parameter("abort_on_pe_failure").value
        )

        self.Tini = int(self.get_parameter("Tini").value)
        self.N = int(self.get_parameter("horizon").value)
        self.Q_diag = list(self.get_parameter("Q_diag").value)
        self.R_diag = list(self.get_parameter("R_diag").value)
        self.lambda_g = float(self.get_parameter("lambda_g").value)
        self.lambda_s = float(self.get_parameter("lambda_s").value)

        self.solver_name = str(self.get_parameter("solver_name").value)
        self.enable_warmup = bool(self.get_parameter("enable_warmup").value)

    def _init_deepc_state(self) -> None:
        self.warned_solve = False
        self.warmup_steps = self.Tini if self.enable_warmup else 0

        self.u_hist: Deque[np.ndarray] = deque(maxlen=self.Tini)
        self.y_hist: Deque[np.ndarray] = deque(maxlen=self.Tini)
        self.pending_history_u: Optional[np.ndarray] = None

        self.pred_rows = []
        self.deepc_enabled = True
        self.deepc = None
        self.dataset_desc = None

    def _load_io_data(self) -> None:
        dataset_paths = resolve_dataset_path(
            dataset_csv=self.dataset_csv,
            dataset_dir=self.dataset_dir,
            dataset_glob=self.dataset_glob,
            dataset_mode=self.dataset_mode,
        )

        self.u_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.mosaic_datasets: Optional[List[dict]] = None

        if self.dataset_mode == "single":
            csv_path = dataset_paths[0]
            self.u_data, self.y_data = load_dataset_csv(
                csv_path,
                drop_initial_rows=self.drop_initial_rows,
                max_rows=self.max_rows_per_dataset,
                yaw_representation=self.yaw_representation,
                y_shift_steps=self.dataset_y_shift_steps,
            )
            self.dataset_desc = csv_path

        elif self.dataset_mode == "mosaic":
            self.mosaic_datasets = load_multiple_dataset_csvs(
                dataset_paths,
                drop_initial_rows=self.drop_initial_rows,
                min_rows_per_dataset=self.min_rows_per_dataset,
                max_rows_per_dataset=self.max_rows_per_dataset,
                yaw_representation=self.yaw_representation,
                y_shift_steps=self.dataset_y_shift_steps,
            )
            self.dataset_desc = (
                f"{len(self.mosaic_datasets)} files from "
                f"{self.dataset_dir}/{self.dataset_glob}"
            )

        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")

    def _check_PE_condition(self) -> None:
        if not self.check_pe_before_start:
            return

        pe_order = self.Tini + self.N + self.pe_order_extra
        pe_info = check_PE_condition(
            order=pe_order,
            u_data=self.u_data,
            mosaic_datasets=self.mosaic_datasets,
            tol=self.pe_rank_tol,
        )

        if pe_info["mode"] == "single":
            self.get_logger().info(
                "PE check(single): "
                f"order={pe_info['order']}, "
                f"T={pe_info['T']}, "
                f"rank={pe_info['rank']}/{pe_info['expected_rank']}, "
                f"n_cols={pe_info['n_cols']}, "
                f"min_T_single={pe_info['min_length_single']}, "
                f"sigma_min={pe_info['sigma_min']:.3e}, "
                f"tol={pe_info['tol']:.3e}"
            )
        else:
            self.get_logger().info(
                "PE check(mosaic): "
                f"order={pe_info['order']}, "
                f"rank={pe_info['rank']}/{pe_info['expected_rank']}, "
                f"n_cols={pe_info['n_cols']}, "
                f"used_datasets={pe_info['used_dataset_count']}, "
                f"sigma_min={pe_info['sigma_min']:.3e}, "
                f"tol={pe_info['tol']:.3e}"
            )

        if pe_info["ok"]:
            return

        msg = (
            f"PE check failed for {pe_info['mode']} Hankel. "
            f"rank(Hu)={pe_info['rank']}/{pe_info['expected_rank']}, "
            f"n_cols={pe_info['n_cols']}, "
            f"order={pe_info['order']}."
        )
        if self.abort_on_pe_failure:
            raise RuntimeError(msg)

        self.deepc_enabled = False
        self.get_logger().warn(msg)
        self.get_logger().warn("DeePC disabled. Baseline tracking only.")

    def _build_solver(self) -> None:
        if not self.deepc_enabled:
            return

        if self.dataset_mode == "single":
            self.deepc = DeePCSolver(
                u_data=self.u_data,
                y_data=self.y_data,
                Tini=self.Tini,
                N=self.N,
                Q_diag=self.Q_diag,
                R_diag=self.R_diag,
                lambda_g=self.lambda_g,
                lambda_s=self.lambda_s,
                solver_name=self.solver_name,
            )
        else:
            self.deepc = DeePCSolver(
                u_data=None,
                y_data=None,
                Tini=self.Tini,
                N=self.N,
                Q_diag=self.Q_diag,
                R_diag=self.R_diag,
                lambda_g=self.lambda_g,
                lambda_s=self.lambda_s,
                solver_name=self.solver_name,
                mosaic_datasets=self.mosaic_datasets,
            )

    def encode_solver_output(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> np.ndarray:
        return np.asarray(
            encode_deepc_output(
                x=x,
                y=y,
                yaw=yaw,
                yaw_representation=self.yaw_representation,
            ),
            dtype=np.float64,
        )

    def reference_output_vector(self, ref: RefPoint) -> List[float]:
        return encode_deepc_output(
            x=ref.x,
            y=ref.y,
            yaw=ref.yaw,
            yaw_representation=self.yaw_representation,
        )

    def build_reference_horizon(self) -> np.ndarray:
        y_refs = []
        last_idx = len(self.ref_traj) - 1

        for k in range(self.N):
            idx = min(self.ref_idx + self.dataset_y_shift_steps + k, last_idx)
            ref_k = self.ref_traj[idx]
            y_refs.append(self.reference_output_vector(ref_k))

        return np.asarray(y_refs, dtype=np.float64).reshape(-1)

    def baseline_tracking_law(
        self,
        ref: RefPoint,
        e_x: float,
        e_y: float,
        e_psi: float,
    ) -> Tuple[float, float]:
        return unicycle_tracking_law(
            ref.v,
            ref.w,
            e_x,
            e_y,
            e_psi,
            kx=self.kx,
            ky=self.ky,
            kpsi=self.kpsi,
            v_min=self.v_min,
            v_max=self.v_max,
            w_min=self.w_min,
            w_max=self.w_max,
        )

    def in_warmup_phase(self) -> bool:
        return self.enable_warmup and self.step_idx < self.warmup_steps

    def update_history_before_solve(self, y_now: np.ndarray) -> None:
        if self.dataset_y_shift_steps != 1:
            return
        if self.pending_history_u is None:
            return
        self.u_hist.append(self.pending_history_u)
        self.y_hist.append(y_now)
        self.pending_history_u = None

    def update_history_after_command(
        self,
        cmd_v: float,
        cmd_w: float,
        y_now: np.ndarray,
    ) -> None:
        u_now = np.asarray([cmd_v, cmd_w], dtype=np.float64)
        if self.dataset_y_shift_steps == 1:
            self.pending_history_u = u_now
            return
        self.u_hist.append(u_now)
        self.y_hist.append(y_now)

    def log_prediction_step(
        self,
        mode: str,
        sim_time: float,
        ref_idx: int,
        u_pred: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        if self.deepc is None:
            return

        u_dim, y_dim = self.get_solver_dims()

        for k in range(self.N):
            u_base = k * u_dim
            y_base = k * y_dim

            row = {
                "step": self.step_idx,
                "mode": mode,
                "sim_time_sec": sim_time,
                "ref_idx": ref_idx,
                "pred_step": k,
                "u_v": float(u_pred[u_base + 0]),
                "u_w": float(u_pred[u_base + 1]),
                "y_x": float(y_pred[y_base + 0]),
                "y_y": float(y_pred[y_base + 1]),
            }
            yaw_pred = float(y_pred[y_base + 2])
            if not self.uses_unwrapped_yaw:
                yaw_pred = wrap_to_pi(yaw_pred)
            row.update({
                "y_yaw": yaw_pred,
            })
            self.pred_rows.append(row)

    def get_solver_dims(self) -> Tuple[int, int]:
        if self.deepc is None:
            raise RuntimeError("Solver is not initialized")
        if hasattr(self.deepc, "u_dim") and hasattr(self.deepc, "y_dim"):
            return int(self.deepc.u_dim), int(self.deepc.y_dim)
        raise RuntimeError("Unable to infer solver dimensions")

    def save_prediction_csv(self) -> None:
        if not self.pred_rows:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(
            self.output_dir,
            f"{self.file_prefix}_prediction_{self.current_run_stamp}.csv",
        )

        fieldnames = [
            "step", "mode", "sim_time_sec",
            "ref_idx", "pred_step",
            "u_v", "u_w",
            "y_x", "y_y", "y_yaw",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.pred_rows)

        self._safe_info(f"Saved prediction log to: {csv_path}")

    def save_additional_tracking_outputs(self) -> None:
        self.save_prediction_csv()

    def control_once(self) -> None:
        if self.finished:
            return

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg)

        if self.last_odom is None:
            self.publish_cmd(0.0, 0.0)
            return

        x, y, yaw, v_meas, w_meas = self.current_measured_state()
        self.append_actual_path(x, y, yaw)

        self.update_reference_index(x, y)
        self.check_abort_condition(x, y)
        if self.finished:
            return

        track_ref = self.get_tracking_ref()

        e_x, e_y, e_psi = self.compute_body_frame_error(track_ref, x, y, yaw)
        y_now = self.encode_solver_output(x, y, yaw)
        self.update_history_before_solve(y_now)

        cmd_v = 0.0
        cmd_w = 0.0
        mode = "deepc"
        u_pred = None
        y_pred = None

        if self.in_warmup_phase():
            cmd_v, cmd_w = self.baseline_tracking_law(track_ref, e_x, e_y, e_psi)
            mode = "warmup"
        else:
            if not self.deepc_enabled or self.deepc is None:
                self.get_logger().error(
                    "DeePC is unavailable. "
                    "Keep abort_on_pe_failure:=true so the node does not continue "
                    "without DeePC."
                )
                self.publish_cmd(0.0, 0.0)
                return

            try:
                u_dim, y_dim = self.get_solver_dims()
                u_ini = stack_history_with_zero_padding(
                    self.u_hist,
                    u_dim,
                    self.Tini,
                )
                y_ini = stack_history_with_zero_padding(
                    self.y_hist,
                    y_dim,
                    self.Tini,
                )

                y_ref = self.build_reference_horizon()
                u0, u_pred, y_pred = self.deepc.solve(
                    u_ini=u_ini,
                    y_ini=y_ini,
                    y_ref=y_ref,
                    u_min=np.asarray([self.v_min, self.w_min], dtype=np.float64),
                    u_max=np.asarray([self.v_max, self.w_max], dtype=np.float64),
                )
                cmd_v = clamp(float(u0[0]), self.v_min, self.v_max)
                cmd_w = clamp(float(u0[1]), self.w_min, self.w_max)
            except Exception as exc:
                cmd_v, cmd_w = self.baseline_tracking_law(
                    track_ref,
                    e_x,
                    e_y,
                    e_psi,
                )
                mode = "baseline"
                if (not self.warned_solve) or (self.step_idx % 100 == 0):
                    self.get_logger().warn(f"DeePC solver fallback to baseline: {exc}")
                    self.warned_solve = True

        sim_time = self.get_clock().now().nanoseconds * 1e-9

        if mode == "deepc" and u_pred is not None and y_pred is not None:
            self.log_prediction_step(
                mode=mode,
                sim_time=sim_time,
                ref_idx=self.ref_idx,
                u_pred=u_pred,
                y_pred=y_pred,
            )

        self.log_step(
            mode=mode,
            ref=track_ref,
            x=x,
            y=y,
            yaw=yaw,
            v_meas=v_meas,
            w_meas=w_meas,
            e_x=e_x,
            e_y=e_y,
            e_psi=e_psi,
            cmd_v=cmd_v,
            cmd_w=cmd_w,
        )

        self.publish_cmd(cmd_v, cmd_w)
        self.update_history_after_command(cmd_v, cmd_w, y_now)

        final_ref = self.ref_traj[-1]
        e_fx, e_fy, e_fpsi = self.compute_body_frame_error(final_ref, x, y, yaw)
        self.check_finish_condition(e_fx, e_fy, e_fpsi)
        if self.finished:
            return

        self.get_logger().info(
            f"[{mode}] step={self.step_idx:4d} ref_idx={self.ref_idx:4d} "
            f"e=({e_x:+.3f}, {e_y:+.3f}, {e_psi:+.3f}) "
            f"u=({cmd_v:+.3f}, {cmd_w:+.3f}) "
            f"meas=({v_meas:+.3f}, {w_meas:+.3f})"
        )

        self.step_idx += 1


def main() -> None:
    rclpy.init()
    node = DeePCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
