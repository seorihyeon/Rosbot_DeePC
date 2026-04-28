#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("/ws/results")


def resolve_csv_path() -> Path:
    # If argument exists:
    # 1) run with argument itself
    # 2) if file noe exists, run with /ws/results/<argument>
    if len(sys.argv) >= 2:
        arg = Path(sys.argv[1])

        if arg.exists():
            return arg.resolve()

        candidate = RESULTS_DIR / sys.argv[1]
        if candidate.exists():
            return candidate.resolve()

        raise FileNotFoundError(
            f"CSV file not found: '{sys.argv[1]}'\n"
            f"- not found in both {sys.argv[1]} and {candidate}"
        )

    # If there is no arguemnt, run with most recent csv file in /ws/results
    csv_files = sorted(
        RESULTS_DIR.glob("*_run_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:
        raise FileNotFoundError(f"There is no csv file in {RESULTS_DIR}.")

    return csv_files[0].resolve()


def main():
    csv_path = resolve_csv_path()
    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["ref_x"], df["ref_y"], label="reference")
    plt.plot(df["x"], df["y"], label="rosbot")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Reference vs Robot Trajectory\n{csv_path.name}")
    plt.show()


if __name__ == "__main__":
    main()