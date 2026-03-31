import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/ws/results/deepc_run_20260330_180326.csv")

plt.figure()
plt.plot(df["ref_x"], df["ref_y"], label="reference")
plt.plot(df["x"], df["y"], label="robot")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Reference vs Robot Trajectory")
plt.show()