import pandas as pd
import numpy as np
from faker import Faker
import streamlit as st
import joblib
from models import RandomForest, DecisionTree, Node

# Tải mô hình và label encoder từ tệp
rf_model = joblib.load('rf_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Khởi tạo Faker để tạo địa chỉ giả
fake = Faker()

# Tải lại DataFrame từ tệp CSV
df = pd.read_csv('sample_transactions.csv')
df['time'] = pd.to_datetime(df['time'])  # Chuyển đổi cột time sang kiểu datetime

# Duy trì danh sách khách hàng bị chặn
blocked_customers = set(df[df['is_fraud'] == 1]['customer_id'].unique())

# Giao diện nhập liệu với Streamlit
st.title("Dự đoán giao dịch gian lận")

amount = st.number_input("Nhập số tiền giao dịch:", min_value=0)
customer_id = st.number_input("Nhập mã khách hàng:", min_value=0)
transaction_type = st.selectbox("Chọn loại giao dịch:", ["online", "in-store"])

# Kiểm tra nếu khách hàng bị chặn
if customer_id in blocked_customers:
    st.error(f"Mã khách hàng {customer_id} đã bị chặn do có giao dịch gian lận trước đó!")
else:
    # Tự động tính số lượng giao dịch trong 1 ngày
    current_time = pd.Timestamp.now()
    one_day_ago = current_time - pd.Timedelta(days=1)

    # Lọc giao dịch trong 1 ngày của khách hàng
    transaction_count_last_1_days = df[df['customer_id'] == customer_id]['transaction_count_last_1_days'].max()

    # Hiển thị số lượng giao dịch tự động tính toán
    st.write(f"Số lượng giao dịch trong 1 ngày: {transaction_count_last_1_days}")

    if st.button("Dự đoán"):
        # Kiểm tra điều kiện gian lận
        is_fraud = 0
        fraud_reason = ""

        if transaction_count_last_1_days > 10:
            is_fraud = 1
            fraud_reason = "Số giao dịch trong 1 ngày quá nhiều."

        if amount > 3000:
            is_fraud = 1
            if fraud_reason:
                fraud_reason += " "
            fraud_reason += "Số tiền giao dịch lớn."

        # Kiểm tra xem customer_id đã tồn tại trong DataFrame chưa
        if customer_id in df['customer_id'].values:
            # Cập nhật thông tin cho khách hàng hiện tại
            df.loc[df['customer_id'] == customer_id, 'transaction_count_last_1_days'] = transaction_count_last_1_days + 1
            df.loc[df['customer_id'] == customer_id, 'amount'] = amount
            df.loc[df['customer_id'] == customer_id, 'time'] = current_time
            df.loc[df['customer_id'] == customer_id, 'is_fraud'] = is_fraud
            df.loc[df['customer_id'] == customer_id, 'fraud_reason'] = fraud_reason.strip()
            df.loc[df['customer_id'] == customer_id, 'transaction_type'] = label_encoder.transform([transaction_type])[0]

            transaction_count_last_1_days += 1
        else:
            # Tạo giao dịch mới nếu chưa tồn tại
            new_transaction = {
                'transaction_id': len(df) + 1,
                'time': current_time,
                'amount': amount,
                'transaction_type': label_encoder.transform([transaction_type])[0],
                'address': fake.address().replace('\n', ', '),
                'customer_id': customer_id,
                'is_fraud': is_fraud,
                'fraud_reason': fraud_reason.strip(),
                'transaction_count_last_1_days': 1,
            }

            # Thêm giao dịch mới vào DataFrame
            df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)

            transaction_count_last_1_days = 1

        # Nếu giao dịch là gian lận, thêm customer_id vào danh sách bị chặn
        if is_fraud == 1:
            blocked_customers.add(customer_id)

        # Lưu DataFrame vào tệp CSV
        df.to_csv('sample_transactions.csv', index=False)
        df = pd.read_csv('sample_transactions.csv')

        st.write(f"Số lượng giao dịch trong 1 ngày (Cập nhập): {transaction_count_last_1_days}")
        
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
            'transaction_count_last_1_days': transaction_count_last_1_days,
        }])
        model_prediction = rf_model.predict(model_input.values)
        model_result = "Gian lận" if model_prediction[0] == 1 else "Không gian lận"
        st.write(f"Kết quả dự đoán từ mô hình: {model_result}")

        # Hiển thị DataFrame cập nhật
        st.subheader("Dữ liệu giao dịch cập nhật")
        st.write(df)
