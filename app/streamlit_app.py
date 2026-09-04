"""Giao diện Web ứng dụng Streamlit cho Phân loại Giọng nói Parkinson.

Cho phép bác sĩ/nghiên cứu viên tải lên tệp CSV chứa 22 đặc trưng âm thanh,
hiển thị bảng kết quả dự đoán ở cấp độ bản ghi và cấp độ bệnh nhân, biểu đồ xác suất
và cho phép xuất tệp kết quả dự đoán dạng CSV.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.settings import ARTIFACT_PATH, RESEARCH_WARNING
from src.data import ORIGINAL_FEATURES
from src.predict import load_bundle, predict_records

# Cấu hình giao diện trang web Streamlit
st.set_page_config(
    page_title="Parkinson Voice Research Demo",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Phân loại Giọng nói Parkinson (Leakage-Aware)")
st.warning(f"⚠️ {RESEARCH_WARNING} Mô hình không phải thiết bị y tế.")

st.markdown(
    "Tải lên tệp CSV chứa cột `name` và 22 đặc trưng tần số/biên độ giọng nói từ bộ dữ liệu UCI. "
    "Mô hình thực hiện dự đoán từng bản ghi âm, sau đó **gộp xác suất theo cấp độ bệnh nhân** "
    "dựa trên quy tắc đã được tối ưu hóa hoàn toàn từ Out-Of-Fold (OOF) Train."
)


with st.expander("📋 Xem Schema tệp CSV bắt buộc"):
    st.code("name, " + ", ".join(ORIGINAL_FEATURES), language=None)

uploaded_file = st.file_uploader("Tải lên tệp CSV dữ liệu giọng nói", type=["csv"])

if uploaded_file is not None:
    try:
        input_frame = pd.read_csv(uploaded_file)
        bundle = load_bundle(ARTIFACT_PATH)
        record_results, subject_results = predict_records(input_frame, bundle)
    except Exception as exc:
        st.error(f"❌ Không thể xử lý tệp: {exc}")
        st.stop()

    st.success(
        f"✅ Đã xử lý thành công {len(record_results)} bản ghi âm của "
        f"{len(subject_results)} bệnh nhân bằng mô hình **{bundle['champion_name']}**."
    )
    st.caption(
        " Quy tắc gộp xác suất được khóa từ OOF Train: "
        f"Phương pháp gộp `{bundle.get('probability_aggregation', 'mean')}`, "
        f"Ngưỡng phân loại `{float(bundle['decision_threshold']):.3f}`."
    )

    st.subheader("📊 Kết quả tổng hợp theo Bệnh nhân (Subject-Level)")
    st.dataframe(
        subject_results.style.format({"probability_status_1": "{:.1%}"}),
        use_container_width=True,
    )

    chart_frame = subject_results.set_index("subject_id")[["probability_status_1"]]
    st.bar_chart(chart_frame, y_label="Xác suất mắc bệnh (status = 1)", horizontal=False)

    st.subheader("📝 Kết quả chi tiết từng Bản ghi âm (Record-Level)")
    st.dataframe(
        record_results.style.format({"probability_status_1": "{:.1%}"}),
        use_container_width=True,
    )

    st.subheader("📈 Trực quan hóa Đặc trưng Giọng nói")
    feature = st.selectbox("Chọn đặc trưng phân tích", bundle["feature_columns"])
    feature_chart = input_frame.assign(subject_id=record_results["subject_id"])
    st.bar_chart(feature_chart, x="subject_id", y=feature, y_label=feature)

    # Nút tải xuống kết quả CSV
    output = io.StringIO()
    subject_results.to_csv(output, index=False)
    st.download_button(
        "📥 Tải tệp kết quả dự đoán CSV",
        data=output.getvalue().encode("utf-8-sig"),
        file_name="parkinsons_subject_predictions.csv",
        mime="text/csv",
    )
