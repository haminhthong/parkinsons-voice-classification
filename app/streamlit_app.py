"""Ứng dụng Streamlit dự đoán CSV theo bệnh nhân."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import ORIGINAL_FEATURES
from src.predict import load_bundle, predict_records

ARTIFACT_PATH = Path("artifacts/parkinsons_champion_pipeline.joblib")

st.set_page_config(page_title="Parkinson Voice Research Demo", page_icon="🎙️", layout="wide")
st.title("Phân loại giọng nói Parkinson")
st.warning("Chỉ phục vụ nghiên cứu, không dùng để chẩn đoán hoặc thay thế tư vấn y khoa.")
st.write(
    "Tải lên CSV gồm cột `name` và 22 đặc trưng giọng nói. "
    "Ứng dụng dự đoán từng bản ghi, sau đó lấy trung bình xác suất theo bệnh nhân."
)

with st.expander("Schema CSV bắt buộc"):
    st.code("name, " + ", ".join(ORIGINAL_FEATURES), language=None)

uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])
if uploaded_file is not None:
    try:
        input_frame = pd.read_csv(uploaded_file)
        bundle = load_bundle(ARTIFACT_PATH)
        record_results, subject_results = predict_records(input_frame, bundle)
    except Exception as exc:  # Streamlit cần chuyển lỗi schema thành thông báo dễ hiểu.
        st.error(f"Không thể xử lý tệp: {exc}")
        st.stop()

    st.success(
        f"Đã xử lý {len(record_results)} bản ghi của "
        f"{len(subject_results)} bệnh nhân bằng {bundle['champion_name']}."
    )
    st.subheader("Kết quả tổng hợp theo bệnh nhân")
    st.dataframe(
        subject_results.style.format({"probability_status_1": "{:.1%}"}),
        use_container_width=True,
    )

    chart_frame = subject_results.set_index("subject_id")[["probability_status_1"]]
    st.bar_chart(chart_frame, y_label="Xác suất status = 1", horizontal=False)

    st.subheader("Kết quả từng bản ghi")
    st.dataframe(
        record_results.style.format({"probability_status_1": "{:.1%}"}),
        use_container_width=True,
    )

    st.subheader("Biểu đồ đặc trưng")
    feature = st.selectbox("Chọn đặc trưng", bundle["feature_columns"])
    feature_chart = input_frame.assign(subject_id=record_results["subject_id"])
    st.bar_chart(feature_chart, x="subject_id", y=feature, y_label=feature)

    output = io.StringIO()
    subject_results.to_csv(output, index=False)
    st.download_button(
        "Tải kết quả dự đoán",
        data=output.getvalue().encode("utf-8-sig"),
        file_name="parkinsons_subject_predictions.csv",
        mime="text/csv",
    )

