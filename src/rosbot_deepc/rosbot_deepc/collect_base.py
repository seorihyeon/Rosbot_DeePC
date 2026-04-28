import csv
import os
from datetime import datetime

import rclpy

from .runtime_base import RuntimeBase


class CollectBase(RuntimeBase):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._declare_collect_parameters()
        self._load_collect_parameters()
        self._init_collect_state()

    def _declare_collect_parameters(self) -> None:
        self.declare_parameter("output_dir", "/ws/datasets")
        self.declare_parameter("file_prefix", "collect")
        self.declare_parameter("warmup_steps", 20)

    def _load_collect_parameters(self) -> None:
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.file_prefix = str(self.get_parameter("file_prefix").value)
        self.warmup_steps = int(self.get_parameter("warmup_steps").value)

    def _init_collect_state(self) -> None:
        self.step_idx = 0
        self.csv_path = ""
        self.csv_file = None
        self.writer = None

        os.makedirs(self.output_dir, exist_ok=True)

    def open_output_csv(self, stem: str, header: list[str]) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            self.output_dir,
            f"{self.file_prefix}_{stem}_{stamp}.csv",
        )
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(header)
        self.csv_file.flush()

    def write_csv_row(self, row: list) -> None:
        self.writer.writerow(row)
        if self.step_idx % 50 == 0:
            self.csv_file.flush()

    def close_output_csv(self) -> None:
        if self.csv_file is None:
            return
        try:
            if not self.csv_file.closed:
                self.csv_file.flush()
                self.csv_file.close()
        finally:
            self.csv_file = None
            self.writer = None

    def finish_and_shutdown(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.cancel_common_timers()
        self.publish_stop_commands()
        self.close_output_csv()
        self.get_logger().info("Finished data collection, shutting down...")
        self.request_shutdown()
