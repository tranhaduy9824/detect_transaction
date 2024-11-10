import pandas as pd
import numpy as np
from faker import Faker
import streamlit as st
import joblib
from collections import Counter
from models import RandomForest, DecisionTree, Node

# Tải mô hình và label encoder từ tệp
rf_model = joblib.load('rf_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Khởi tạo Faker để tạo địa chỉ giả
fake = Faker()

# Tải lại DataFrame từ tệp CSV
df = pd.read_csv('sample_transactions.csv')

# Giao diện nhập liệu với Streamlit
st.title("Dự đoán giao dịch gian lận")

amount = st.number_input("Nhập số tiền giao dịch:", min_value=0)
customer_id = st.number_input("Nhập mã khách hàng:", min_value=0)
transaction_type = st.selectbox("Chọn loại giao dịch:", ["online", "in-store"])
transaction_count_last_7_days = st.number_input("Nhập số lượng giao dịch trong 7 ngày qua:", min_value=0)

if st.button("Dự đoán"):
    # Kiểm tra điều kiện gian lận
    is_fraud = 0
    fraud_reason = ""

    if transaction_count_last_7_days > 10:
        is_fraud = 1
        fraud_reason = "Số giao dịch trong 7 ngày quá nhiều."

    if amount > 3000:
        is_fraud = 1
        if fraud_reason:
            fraud_reason += " "
        fraud_reason += "Số tiền giao dịch lớn."

    # Tạo giao dịch mới
    new_transaction = {
        'transaction_id': len(df) + 1,
        'time': pd.Timestamp.now(),
        'amount': amount,
        'transaction_type': label_encoder.transform([transaction_type])[0],
        'address': fake.address().replace('\n', ', '),
        'customer_id': customer_id,
        'is_fraud': is_fraud,
        'fraud_reason': fraud_reason.strip(),
        'transaction_count_last_7_days': transaction_count_last_7_days,
    }

    # Thêm giao dịch mới vào DataFrame
    df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)

    # Lưu DataFrame vào tệp CSV
    df.to_csv('sample_transactions.csv', index=False)

    # Hiển thị kết quả dự đoán và lý do gian lận
    result = "Gian lận" if is_fraud == 1 else "Không gian lận"
    st.write(f"Kết quả dự đoán: {result}")
    if is_fraud:
        st.write(f"Lý do gian lận: {fraud_reason.strip()}")

    # Sử dụng mô hình để dự đoán
    model_input = pd.DataFrame([{
        'amount': amount,
        'customer_id': customer_id,
        'transaction_type': label_encoder.transform([transaction_type])[0],
        'transaction_count_last_7_days': transaction_count_last_7_days,
    }])
    model_prediction = rf_model.predict(model_input.values)
    model_result = "Gian lận" if model_prediction[0] == 1 else "Không gian lận"
    st.write(f"Kết quả dự đoán từ mô hình: {model_result}")

    # Hiển thị DataFrame cập nhật
    st.subheader("Dữ liệu giao dịch cập nhật")
    st.write(df)