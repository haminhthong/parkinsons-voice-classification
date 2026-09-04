"""Module trực quan hóa kết quả và sinh biểu đồ báo cáo.

Tự động đọc các tệp dữ liệu báo cáo từ thư mục `artifacts/` và xuất ra 4 biểu đồ hình ảnh
dùng cho tài liệu minh họa (README.md / Portfolio / Presentation):
1. `model_benchmark.png`: Biểu đồ so sánh F1-macro CV giữa các mô hình.
2. `holdout_probabilities.png`: Biểu đồ xác suất dự đoán trên tập Holdout Test theo bệnh nhân.
3. `feature_selection_stability.png`: Biểu đồ độ ổn định tần suất lựa chọn đặc trưng.
4. `threshold_aggregation.png`: Biểu đồ đường quét ngưỡng quyết định và quy tắc gộp xác suất.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Lưu đồ họa Matplotlib với cấu hình chuẩn dpi=180 và giải phóng bộ nhớ.

    Args:
        figure: Đối tượng plt.Figure.
        path: Đường dẫn tệp ảnh đầu ra.
    """
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_portfolio_figures(
    artifact_dir: str | Path = "artifacts",
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    """Sinh toàn bộ 4 biểu đồ chất lượng cao trực tiếp từ các artifact đã được huấn luyện.

    Args:
        artifact_dir: Thư mục chứa các tệp artifact CSV và JSON (mặc định 'artifacts').
        output_dir: Thư mục lưu xuất các biểu đồ ảnh PNG (mặc định 'reports/figures').

    Returns:
        list[Path]: Danh sách các đường dẫn tệp ảnh đã sinh thành công.
    """
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu từ artifact đã kiểm chứng
    benchmark = pd.read_csv(artifact_dir / "model_benchmark.csv")
    holdout = pd.read_csv(artifact_dir / "holdout_subject_predictions.csv")
    stability = pd.read_csv(artifact_dir / "feature_selection_stability.csv")
    threshold_search = pd.read_csv(artifact_dir / "oof_threshold_search.csv")
    raw_metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    decision_threshold = float(raw_metrics.get("calibration", {}).get("threshold", raw_metrics.get("decision_threshold", 0.5)))

    paths: list[Path] = []

    # Biểu đồ 1: Benchmark các mô hình ứng viên
    benchmark_plot = benchmark.sort_values("Subject F1-macro mean")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(
        benchmark_plot["Subject F1-macro mean"],
        benchmark_plot["Model"],
        xerr=benchmark_plot["Subject F1-macro std"],
        fmt="o",
        color="#1f5f8b",
        ecolor="#9ecae1",
        capsize=4,
    )
    axis.set(xlim=(0, 1), xlabel="F1-macro CV trung bình ± 1 độ lệch chuẩn")
    axis.set_title("Benchmark không rò rỉ theo bệnh nhân")
    axis.grid(axis="x", alpha=0.25)
    paths.append(output_dir / "model_benchmark.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 2: Xác suất Holdout theo bệnh nhân
    holdout_plot = holdout.sort_values("probability")
    colors = holdout_plot["status"].map({0: "#2ca25f", 1: "#de2d26"})
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.scatter(
        holdout_plot["subject_id"],
        holdout_plot["probability"],
        c=colors,
        s=85,
    )
    axis.axhline(
        decision_threshold,
        color="#222222",
        linestyle="--",
        label=f"Ngưỡng OOF = {decision_threshold:.3f}",
    )
    axis.set(
        ylim=(0, 1),
        xlabel="Bệnh nhân holdout",
        ylabel="Xác suất status = 1",
    )
    axis.set_title("Xác suất holdout theo bệnh nhân")
    axis.tick_params(axis="x", rotation=35)
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2ca25f",
            label="Nhãn thật = 0",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#de2d26",
            label="Nhãn thật = 1",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            label=f"Ngưỡng OOF = {decision_threshold:.3f}",
        ),
    ]
    axis.legend(handles=legend_items)
    axis.grid(axis="y", alpha=0.25)
    paths.append(output_dir / "holdout_probabilities.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 3: Độ ổn định khi chọn đặc trưng qua các fold
    stability_plot = stability.head(15).sort_values("Selection frequency")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        stability_plot["Feature"],
        stability_plot["Selection frequency"],
        color="#4c78a8",
    )
    axis.set(xlim=(0, 1), xlabel="Tỷ lệ fold lựa chọn")
    axis.set_title("Độ ổn định khi chọn đặc trưng")
    axis.grid(axis="x", alpha=0.25)
    paths.append(output_dir / "feature_selection_stability.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 4: So sánh quy tắc gộp xác suất và quét ngưỡng quyết định OOF
    figure, axis = plt.subplots(figsize=(9, 5))
    for aggregation, group in threshold_search.groupby("Aggregation"):
        ordered = group.sort_values("Threshold")
        axis.plot(
            ordered["Threshold"],
            ordered["Balanced Accuracy"],
            label=aggregation,
        )
    axis.axvline(
        decision_threshold,
        color="#222222",
        linestyle="--",
        label="Ngưỡng được chọn",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Threshold",
        ylabel="Balanced Accuracy OOF",
    )
    axis.set_title("So sánh threshold và cách gộp xác suất")
    axis.legend()
    axis.grid(alpha=0.25)
    paths.append(output_dir / "threshold_aggregation.png")
    _save_figure(figure, paths[-1])

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo biểu đồ portfolio từ artifact.")
    parser.add_argument("--artifacts", default="artifacts", help="Thư mục chứa artifact.")
    parser.add_argument("--output", default="reports/figures", help="Thư mục xuất ảnh.")
    arguments = parser.parse_args()
    for figure_path in create_portfolio_figures(arguments.artifacts, arguments.output):
        print(figure_path)
