"""Kịch bản kiểm thử tải (Load Test) Locust cho REST API suy luận."""

from pathlib import Path

from locust import HttpUser, between, task

FIXTURE_PATH = Path(__file__).parents[1] / "tests" / "fixtures" / "inference_valid.csv"
SAMPLE_BYTES = FIXTURE_PATH.read_bytes() if FIXTURE_PATH.is_file() else b""


class PredictionUser(HttpUser):
    """User thực hiện gửi request liên tục tới endpoint /predict."""

    wait_time = between(0.5, 2)

    @task
    def predict(self):
        """Gửi request POST /predict kèm tệp CSV dữ liệu giọng nói."""
        self.client.post(
            "/predict",
            files={
                "file": (
                    "inference.csv",
                    SAMPLE_BYTES,
                    "text/csv",
                )
            },
        )
