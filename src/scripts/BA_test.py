#!/usr/bin/env python3
from pathlib import Path
import sys
from collections import deque

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "rosbot_deepc"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from rosbot_deepc.utils import load_dataset_csv, block_hankel

u_data, y_data = load_dataset_csv(
    csv_path="/ws/datasets/eight.csv",
    drop_initial_rows=20,
    max_rows=0,
    yaw_representation="wrap",
    y_shift_steps=0,
)

print(u_data); print(y_data)

Tini = 6; N = 10
u_dim = u_data.shape[0]
y_dim = y_data.shape[0]

Hu = block_hankel(u_data, Tini+N)
Hy = block_hankel(y_data, Tini+N)

print(f"Hu: ({Hu.shape[0]}, {Hu.shape[1]})")
print(f"Hy: ({Hy.shape[0]}, {Hy.shape[1]})")

#print(Hu); print(Hy)

Up = Hu[: u_dim*Tini, :]
Uf = Hu[u_dim*Tini :, :]
Yp = Hy[: y_dim*Tini, :]
Yf = Hy[y_dim*Tini :, :]

u_test, y_test = load_dataset_csv(
    csv_path="/ws/datasets/eight.csv",
    drop_initial_rows=20,
    max_rows=0,
    yaw_representation="wrap",
    y_shift_steps=0,
)

anchor = 23

u_ini = u_test[:, anchor-Tini:anchor]
y_ini = y_test[:, anchor-Tini:anchor]
u_f = u_test[:, anchor:anchor+N]
y_a = y_test[:, anchor:anchor+N]

u_ini = np.asarray(u_ini, dtype=np.float64).reshape(-1)
y_ini = np.asarray(y_ini, dtype=np.float64).reshape(-1)
u_f = np.asarray(u_f, dtype=np.float64).reshape(-1)
y_a = np.asarray(y_a, dtype=np.float64).reshape(-1)

#print(u_ini); print(u_f)
print(u_ini.shape)
print(y_ini.shape)
print(u_f.shape)

sim_vector = np.concatenate([u_ini, u_f, y_ini])
print(sim_vector.shape)

H_temp = np.vstack((Up, Uf, Yp))
H_temp_pinv = np.linalg.pinv(H_temp)

pinv_g = H_temp_pinv @ sim_vector

y_f = Yf @ pinv_g

print(y_a)
print(y_f)
