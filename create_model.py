import pandas as pd
import numpy as np
from faker import Faker
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter
from models import RandomForest, DecisionTree, Node

# Khởi tạo Faker để tạo địa chỉ giả
fake = Faker()

# Tạo dữ liệu với các tiêu chí cụ thể
np.random.seed(42)
n_transactions = 1000  # Tăng số lượng giao dịch để có dữ liệu đa dạng hơn

# Tạo số tiền giao dịch và gán thêm yếu tố gian lận hợp lý
amounts = np.random.randint(50, 10000, size=n_transactions)

# Giả sử giao dịch gian lận thường có số tiền lớn hoặc số giao dịch cao trong 1 ngày qua
fraud_labels = []
fraud_reasons = []
transaction_count_last_1_days = np.random.choice([1, 5, 10, 15, 20], size=n_transactions)

# Tạo nhãn gian lận dựa trên các tiêu chí:
for i in range(n_transactions):
    amount = amounts[i]
    count_last_1_days = transaction_count_last_1_days[i]
    
    # Mặc định là không gian lận và không có lý do gian lận
    is_fraud = 0
    fraud_reason = ""

    # Kiểm tra số lượng giao dịch trong 1 ngày qua
    if count_last_1_days > 10:
        is_fraud = 1
        fraud_reason = "Số giao dịch trong 1 ngày quá nhiều."

    # Kiểm tra số tiền giao dịch
    if amount > 3000:
        is_fraud = 1
        if fraud_reason:
            fraud_reason += " "
        fraud_reason += "Số tiền giao dịch lớn."

    fraud_labels.append(is_fraud)
    fraud_reasons.append(fraud_reason)

# Tạo DataFrame
data = {
    'transaction_id': range(1, n_transactions + 1),
    'time': pd.date_range(start='2024-01-01', periods=n_transactions, freq='h'),
    'amount': amounts,
    'transaction_type': np.random.choice(['online', 'in-store'], size=n_transactions),
    'address': [fake.address().replace('\n', ', ') for _ in range(n_transactions)],
    'customer_id': np.random.randint(10000, 20000, size=n_transactions),
    'is_fraud': fraud_labels,
    'fraud_reason': fraud_reasons,
    'transaction_count_last_1_days': transaction_count_last_1_days,
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Chuyển đổi cột transaction_type thành số
label_encoder = LabelEncoder()
df['transaction_type'] = label_encoder.fit_transform(df['transaction_type'])

# Huấn luyện mô hình dự đoán
X = df[['amount', 'customer_id', 'transaction_type', 'transaction_count_last_1_days']].values
y = df['is_fraud'].values
rf_model = RandomForest(n_trees=10, max_depth=10)
rf_model.fit(X, y)

# Lưu mô hình vào tệp
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')