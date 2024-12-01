import pandas as pd
import numpy as np
import streamlit as st
import joblib
from datetime import datetime
import json

# Tải mô hình và label encoder từ tệp
rf_model = joblib.load('rf_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Tải lại DataFrame từ tệp CSV
df = pd.read_csv('sample_transactions.csv')
df['time'] = pd.to_datetime(df['time'])

# Duy trì danh sách khách hàng bị chặn
blocked_customers = set(df[df['is_abnormal'] == 1]['customer_id'].unique())

# Giao diện nhập liệu với Streamlit
st.title("Dự đoán giao dịch gian lận")

amount = st.number_input("Nhập số tiền giao dịch:", min_value=0)
customer_id = st.number_input("Nhập mã khách hàng:", min_value=0)

# Kiểm tra nếu khách hàng bị chặn
if customer_id in blocked_customers:
    st.error(f"Mã khách hàng {customer_id} đã bị chặn do có giao dịch gian lận trước đó!")
else:
    current_time = pd.Timestamp.now()
    
    transaction_count_last_1_days = df.loc[df['customer_id'] == customer_id, 'transaction_count_last_1_days']
    transaction_count_last_1_days = transaction_count_last_1_days.values[0] if len(transaction_count_last_1_days) > 0 else 0

    # Hiển thị số lượng giao dịch tự động tính toán
    st.write(f"Số lượng giao dịch trong 1 ngày: {transaction_count_last_1_days}")

    if st.button("Dự đoán"):
        # Xử lý rolling_history
        rolling_history = []
        if customer_id in df['customer_id'].values:
            rolling_history_json = df.loc[df['customer_id'] == customer_id, 'rolling_history'].values[0]
            rolling_history = json.loads(rolling_history_json) if isinstance(rolling_history_json, str) else []
        
        # Thêm giao dịch hiện tại vào rolling_history
        current_transaction = [amount, current_time.strftime("%Y-%m-%d %H:%M:%S")]
        rolling_history.append(current_transaction)

        # Debug: In rolling history ra để kiểm tra
        st.write("Cập nhật Rolling History:", rolling_history)

        # Tính toán thống kê từ rolling_history
        def process_rolling_history(rolling_history):
            try:
                history = [float(tx[0]) for tx in rolling_history]
                return np.mean(history), np.std(history)
            except:
                return 0.0, 0.0

        rolling_mean, rolling_std = process_rolling_history(rolling_history)

        # Tạo input cho mô hình
        model_input = pd.DataFrame([{
            'amount': float(amount),
            'customer_id': int(customer_id),
            'transaction_count_last_1_days': int(transaction_count_last_1_days),
            'rolling_history_mean': rolling_mean,
            'rolling_history_std': rolling_std
        }])

        # Dự đoán với mô hình
        try:
            model_prediction = rf_model.predict(model_input.values)
            model_result = "Gian lận" if model_prediction[0] == 1 else "Không gian lận"
            st.write(f"Kết quả dự đoán từ mô hình: {model_result}")

            if model_prediction[0] == 1:
                blocked_customers.add(customer_id)
                st.warning("Khách hàng đã bị thêm vào danh sách chặn.")
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

        if customer_id in df['customer_id'].values:
            # Cập nhật dòng tương ứng
            df.loc[df['customer_id'] == customer_id, 'is_abnormal'] = model_prediction[0]
            df.loc[df['customer_id'] == customer_id, 'rolling_history'] = json.dumps(rolling_history)
            df.loc[df['customer_id'] == customer_id, 'transaction_count_last_1_days'] = len(rolling_history)
            df.loc[df['customer_id'] == customer_id, 'time'] = current_time
            df.loc[df['customer_id'] == customer_id, 'amount'] = amount
        else:
            # Thêm dòng mới nếu không tồn tại
            new_transaction = {
                'transaction_id': len(df) + 1,
                'time': current_time,
                'amount': amount,
                'customer_id': customer_id,
                'is_abnormal': model_prediction[0],
                'transaction_count_last_1_days': transaction_count_last_1_days + 1,
                'rolling_history': json.dumps(rolling_history)
            }
            df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)
            
        df.to_csv('sample_transactions.csv', index=False)

        # Hiển thị DataFrame cập nhật
        st.subheader("Dữ liệu giao dịch cập nhật")
        st.write(df)
