"""Sinh notebook Colab tự chứa để tái lập kết quả repository."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_colab_reproducible.ipynb"
INCLUDED_FILES = [
    "configs/default.json",
    "data/parkinsons.csv",
    *[str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "src").glob("*.py")],
]


def make_payload() -> str:
    """Nén mã nguồn và dữ liệu tối thiểu vào một chuỗi base64."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in INCLUDED_FILES:
            archive.write(ROOT / relative_path, relative_path)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def py(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip(),
    }


def build() -> Path:
    payload = make_payload()
    expected_metrics = (ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
        "accelerator": "CPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
    }
    notebook["cells"] = [
        md("""
# Leakage-Aware Parkinson’s Voice Classification — Google Colab

Notebook tự chứa này chạy đúng code của repository: kiểm tra 22 đặc trưng, chia theo bệnh nhân,
benchmark 6 mô hình, calibration theo nhóm, chọn cách gộp/ngưỡng trên OOF train và đánh giá holdout.

**Kết quả chuẩn:** champion `KNN + sigmoid calibration`, gộp `max`, ngưỡng `0.835`,
holdout Accuracy `0.875`, Balanced Accuracy `0.750`, ROC-AUC `0.750`.

> Chỉ phục vụ nghiên cứu và học tập, không dùng để chẩn đoán. Chọn **Runtime → Run all**.
"""),
        md("## 1. Khóa môi trường chạy"),
        py("""
import os, subprocess, sys
IN_COLAB = "google.colab" in sys.modules
PACKAGES = [
    "pandas==2.2.3", "numpy==2.1.3", "scikit-learn==1.5.2",
    "joblib==1.4.2", "matplotlib==3.9.2", "seaborn==0.13.2",
]
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
print("Môi trường:", "Google Colab" if IN_COLAB else "Python cục bộ")
"""),
        md("""
## 2. Khôi phục dự án tối thiểu

Dữ liệu, cấu hình và các module `src` được nhúng trong notebook nên không phụ thuộc Google Drive,
đường dẫn Windows hoặc GitHub. Đây là bản chụp của code repository tại thời điểm tạo notebook.
"""),
        py(f"""
import base64, io, shutil, zipfile
from pathlib import Path

PAYLOAD = "{payload}"
PROJECT_DIR = (
    Path("/content/parkinsons-voice-classification")
    if IN_COLAB
    else Path.cwd() / "parkinsons-colab-runtime"
)
if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)
PROJECT_DIR.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(PROJECT_DIR)
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print("Thư mục chạy:", PROJECT_DIR)
"""),
        md("## 3. Kiểm tra phiên bản, checksum và schema"),
        py("""
import joblib, matplotlib, numpy as np, pandas as pd, sklearn
from src.data import ORIGINAL_FEATURES, SUBJECT_COLUMN, TARGET_COLUMN, load_data
from src.utils import sha256_file

expected_versions = {
    "pandas": "2.2.3", "numpy": "2.1.3", "scikit-learn": "1.5.2",
    "joblib": "1.4.2", "matplotlib": "3.9.2",
}
actual_versions = {
    "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
    "joblib": joblib.__version__, "matplotlib": matplotlib.__version__,
}
if IN_COLAB:
    assert actual_versions == expected_versions, (actual_versions, expected_versions)
DATA_PATH = Path("data/parkinsons.csv")
assert sha256_file(DATA_PATH) == "32e6040916d2f5b80b49589d925a92bd25420687c76be19d72e37205e104abe6"
frame = load_data(DATA_PATH)
assert len(ORIGINAL_FEATURES) == 22
assert set(frame[TARGET_COLUMN].unique()) == {0, 1}
assert frame[SUBJECT_COLUMN].nunique() == 32
print(f"✅ {len(frame)} bản ghi | 32 bệnh nhân | 22 đặc trưng | checksum đúng")
display(frame.head(3))
"""),
        md("## 4. Audit chống rò rỉ bệnh nhân"),
        py("""
from src.data import subject_holdout_split
from src.evaluate import make_subject_folds

train_frame, test_frame = subject_holdout_split(frame, test_size=0.25, random_state=42)
train_ids, test_ids = set(train_frame[SUBJECT_COLUMN]), set(test_frame[SUBJECT_COLUMN])
assert train_ids.isdisjoint(test_ids)
folds = make_subject_folds(train_frame, n_splits=5, random_state=42)
for number, (fit_index, valid_index) in enumerate(folds, 1):
    fit_ids = set(train_frame.iloc[fit_index][SUBJECT_COLUMN])
    valid_ids = set(train_frame.iloc[valid_index][SUBJECT_COLUMN])
    assert fit_ids.isdisjoint(valid_ids)
    assert train_frame.iloc[valid_index][TARGET_COLUMN].nunique() == 2
    print(f"Fold {number}: không overlap, validation đủ hai lớp")
print(f"✅ Holdout: train={len(train_ids)}, test={len(test_ids)}, overlap=0")
"""),
        md("""
## 5. Huấn luyện và benchmark

Scaler và SelectKBest nằm trong Pipeline. Holdout không tham gia chọn champion, calibration,
quy tắc gộp hoặc threshold. Cell này có thể mất vài phút trên CPU Colab.
"""),
        py("""
from src.train import train
ARTIFACT_DIR = Path("artifacts")
benchmark = train(DATA_PATH, ARTIFACT_DIR)
display(benchmark[[
    "Model", "Subject F1-macro mean", "Subject F1-macro std",
    "Subject Balanced Accuracy mean", "Subject ROC-AUC mean",
]])
"""),
        md("## 6. Assert kết quả trùng repository"),
        py(f"""
import json
EXPECTED = json.loads({json.dumps(expected_metrics, ensure_ascii=False)})
ACTUAL = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
assert ACTUAL["selection"]["champion"] == EXPECTED["selection"]["champion"]
assert ACTUAL["holdout_subject"]["Accuracy"] == EXPECTED["holdout_subject"]["Accuracy"]
assert np.isclose(ACTUAL["holdout_subject"]["Balanced Accuracy"], EXPECTED["holdout_subject"]["Balanced Accuracy"], rtol=0, atol=1e-12)
print("✅ KHỚP HOÀN TOÀN VỚI KẾT QUẢ REPOSITORY")
print(json.dumps(ACTUAL, ensure_ascii=False, indent=2))
"""),
        md("## 7. Holdout, khoảng tin cậy và biểu đồ"),
        py("""
from src.report import create_portfolio_figures

display(pd.read_csv(ARTIFACT_DIR / "holdout_subject_predictions.csv"))
display(pd.read_csv(ARTIFACT_DIR / "holdout_bootstrap_ci.csv"))
figure_paths = create_portfolio_figures(ARTIFACT_DIR, Path("reports/figures"))
if IN_COLAB:
    from IPython.display import Image
    for figure_path in figure_paths:
        display(Image(filename=str(figure_path), width=850))
else:
    print("Biểu đồ:", *figure_paths, sep="\\n- ")
"""),
        md("""
## 8. Dự đoán CSV mới — tùy chọn

CSV phải có cột `name` và đủ 22 đặc trưng. Không nhập đặc trưng thủ công.
"""),
        py("""
from src.predict import load_bundle, predict_records
if IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    if uploaded:
        name = next(iter(uploaded))
        inference_frame = pd.read_csv(io.BytesIO(uploaded[name]))
        bundle = load_bundle(ARTIFACT_DIR / "parkinsons_calibrated_pipeline.joblib")
        record_results, subject_results = predict_records(inference_frame, bundle)
        display(subject_results)
        subject_results.to_csv("parkinsons_subject_predictions.csv", index=False)
        files.download("parkinsons_subject_predictions.csv")
else:
    print("Cell upload chỉ kích hoạt trên Google Colab.")
"""),
        md("""
## Kết luận

Khi xuất hiện dòng **KHỚP HOÀN TOÀN**, notebook đã dùng cùng dữ liệu, code, seed và phiên bản
thư viện để tái lập repository. Trong CV, nên nhấn mạnh việc sửa group leakage và công bố bất định
do cỡ mẫu nhỏ thay vì chỉ nêu Accuracy.
"""),
    ]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
