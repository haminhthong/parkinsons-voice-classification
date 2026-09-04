"""Module tạo báo cáo audit về rò rỉ dữ liệu khi phân chia ngẫu nhiên ở cấp bản ghi (Naive Record Split).

Thực hiện phân chia bản ghi ngẫu nhiên (ngược với phân chia theo bệnh nhân), huấn luyện mô hình
và ghi lại tỷ lệ rò rỉ bệnh nhân giữa train và test cùng độ chính xác cao ảo (overoptimistic accuracy).
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data import TARGET_COLUMN, load_data
from src.features import MODEL_FEATURES

DATA_PATH = Path(__file__).parents[1] / "data" / "parkinsons.csv"
ARTIFACT_DIR = Path(__file__).parents[1] / "artifacts"


def run_naive_split_audit(
    data_path: str | Path = DATA_PATH,
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> dict:
    """Thực hiện naive record-level split audit và xuất kết quả vào thư mục artifact."""
    frame = load_data(data_path)

    X = frame[MODEL_FEATURES]
    y = frame[TARGET_COLUMN]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        frame.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_subjects = set(frame.loc[train_idx, "subject_id"])
    test_subjects = set(frame.loc[test_idx, "subject_id"])
    overlapping = train_subjects.intersection(test_subjects)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = float(model.score(X_test, y_test))

    predictions = frame.loc[test_idx, ["name", "subject_id", TARGET_COLUMN]].copy()
    predictions["predicted_status"] = model.predict(X_test)
    predictions["is_overlapping_subject"] = predictions["subject_id"].isin(overlapping)

    audit_metrics = {
        "split_unit": "record",
        "test_size": 0.2,
        "random_state": 42,
        "model": "RandomForestClassifier",
        "test_subjects": len(test_subjects),
        "overlapping_test_subjects": len(overlapping),
        "accuracy": accuracy,
    }

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "naive_split_audit.json").write_text(
        json.dumps(audit_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(output_dir / "naive_split_predictions.csv", index=False)

    return audit_metrics


if __name__ == "__main__":
    metrics = run_naive_split_audit()
    print(json.dumps(metrics, indent=2))
