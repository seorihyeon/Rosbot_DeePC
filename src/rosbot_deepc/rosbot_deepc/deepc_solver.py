from typing import List, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None

from .utils import block_hankel, build_mosaic_hankel


class DeePCSolver:
    @staticmethod
    def _make_diag(values: List[float], expected_dim: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size != expected_dim:
            raise ValueError(f"{name} length mismatch: got {arr.size}, expected {expected_dim}")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be elementwise nonnegative")
        return np.diag(arr)

    def __init__(
        self,
        u_data: Optional[np.ndarray],
        y_data: Optional[np.ndarray],
        Tini: int,
        N: int,
        Q_diag: List[float],
        R_diag: List[float],
        lambda_g: float,
        lambda_s: float,
        solver_name: str = "OSQP",
        mosaic_datasets: Optional[list[dict]] = None,
    ) -> None:
        if cp is None:
            raise RuntimeError("cvxpy is not installed")

        self.Tini = int(Tini)
        self.N = int(N)
        self.L = self.Tini + self.N
        self.lambda_g = float(lambda_g)
        self.lambda_s = float(lambda_s)
        self.Q_diag = list(Q_diag)
        self.R_diag = list(R_diag)
        self.solver_name = solver_name

        self._prepare_data(u_data, y_data, mosaic_datasets)
        self._build_problem()

    def _prepare_data(
        self,
        u_data: Optional[np.ndarray],
        y_data: Optional[np.ndarray],
        mosaic_datasets: Optional[list[dict]],
    ) -> None:
        if mosaic_datasets is not None:
            if len(mosaic_datasets) == 0:
                raise ValueError("mosaic_datasets is empty")
            first_u = mosaic_datasets[0]["u_data"]
            first_y = mosaic_datasets[0]["y_data"]
            self.u_dim = first_u.shape[0]
            self.y_dim = first_y.shape[0]
            Hu, Hy = build_mosaic_hankel(mosaic_datasets, self.L)
        else:
            if u_data is None or y_data is None:
                raise ValueError("u_data and y_data must be provided in single-dataset mode.")
            self.u_dim = u_data.shape[0]
            self.y_dim = y_data.shape[0]
            Hu = block_hankel(u_data, self.L)
            Hy = block_hankel(y_data, self.L)

        self.Up = Hu[: self.u_dim * self.Tini, :]
        self.Uf = Hu[self.u_dim * self.Tini :, :]
        self.Yp = Hy[: self.y_dim * self.Tini, :]
        self.Yf = Hy[self.y_dim * self.Tini :, :]
        self.n_col = Hu.shape[1]

        Qy = self._make_diag(self.Q_diag, expected_dim=self.y_dim, name="Q_diag")
        R = self._make_diag(self.R_diag, expected_dim=self.u_dim, name="R_diag")
        self.Qy_blk = np.kron(np.eye(self.N), Qy)
        self.R_blk = np.kron(np.eye(self.N), R)

    def _build_problem(self) -> None:
        self.u_ini_p = cp.Parameter(self.u_dim * self.Tini)
        self.y_ini_p = cp.Parameter(self.y_dim * self.Tini)
        self.y_ref_p = cp.Parameter(self.y_dim * self.N)
        self.u_min_p = cp.Parameter(self.u_dim)
        self.u_max_p = cp.Parameter(self.u_dim)

        self.g = cp.Variable(self.n_col)
        self.sigma_y = cp.Variable(self.y_dim * self.Tini)
        self.u_f = cp.Variable(self.u_dim * self.N)
        self.y_f = cp.Variable(self.y_dim * self.N)

        cost = (
            cp.quad_form(self.y_f - self.y_ref_p, self.Qy_blk)
            + cp.quad_form(self.u_f, self.R_blk)
            + self.lambda_s * cp.sum_squares(self.sigma_y)
            + self.lambda_g * cp.sum_squares(self.g)
        )

        constraints = [
            self.Up @ self.g == self.u_ini_p,
            self.Yp @ self.g == self.y_ini_p + self.sigma_y,
            self.Uf @ self.g == self.u_f,
            self.Yf @ self.g == self.y_f,
        ]

        for k in range(self.N):
            uk = self.u_f[k * self.u_dim : (k + 1) * self.u_dim]
            constraints += [uk >= self.u_min_p, uk <= self.u_max_p]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        y_ref: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_ini = np.asarray(u_ini, dtype=np.float64).reshape(-1)
        y_ini = np.asarray(y_ini, dtype=np.float64).reshape(-1)
        y_ref = np.asarray(y_ref, dtype=np.float64).reshape(-1)
        u_min = np.asarray(u_min, dtype=np.float64).reshape(-1)
        u_max = np.asarray(u_max, dtype=np.float64).reshape(-1)

        if u_ini.size != self.u_dim * self.Tini:
            raise ValueError(
                f"u_ini length mismatch: got {u_ini.size}, expected {self.u_dim * self.Tini}"
            )
        if y_ini.size != self.y_dim * self.Tini:
            raise ValueError(
                f"y_ini length mismatch: got {y_ini.size}, expected {self.y_dim * self.Tini}"
            )
        if y_ref.size != self.y_dim * self.N:
            raise ValueError(
                f"y_ref length mismatch: got {y_ref.size}, expected {self.y_dim * self.N}"
            )
        if u_min.size != self.u_dim or u_max.size != self.u_dim:
            raise ValueError(f"u_min/u_max length mismatch: expected {self.u_dim}")
        if np.any(u_min > u_max):
            raise ValueError("u_min must be <= u_max elementwise")

        self.u_ini_p.value = u_ini
        self.y_ini_p.value = y_ini
        self.y_ref_p.value = y_ref
        self.u_min_p.value = u_min
        self.u_max_p.value = u_max

        solver = getattr(cp, self.solver_name, None)
        if solver is None:
            raise ValueError(f"Unknown cvxpy solver_name: {self.solver_name}")

        self.problem.solve(solver=solver, warm_start=False, verbose=False)

        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"DeePC solve failed: {self.problem.status}")

        u0 = np.asarray(self.u_f.value[: self.u_dim], dtype=np.float64).copy()
        u_pred = np.asarray(self.u_f.value, dtype=np.float64).copy()
        y_pred = np.asarray(self.y_f.value, dtype=np.float64).copy()
        return u0, u_pred, y_pred
