import pandas as pd
import numpy as np
from faker import Faker
import sys

# Cấu hình lại mã hóa đầu ra của Python thành UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Khởi tạo Faker để tạo địa chỉ giả
fake = Faker()

# Tạo dữ liệu với các tiêu chí cụ thể
np.random.seed(42)
n_transactions = 30  # Số lượng giao dịch

# Tạo số tiền giao dịch và gán thêm yếu tố gian lận hợp lý
amounts = [
    100, 150, 5000, 250, 3000, 200, 800, 450, 1000, 600,  
    650, 2000, 700, 750, 800, 850, 900, 950, 3000, 1000,
    2000, 3000, 150, 1250, 1350, 4000, 1450, 1500, 
    1550, 5000, 1200  # Đảm bảo số lượng phần tử trùng với n_transactions
]

# Giả sử giao dịch gian lận thường có số tiền lớn hoặc số giao dịch cao trong 7 ngày qua
fraud_labels = []
fraud_reasons = []
transaction_count_last_7_days = np.random.choice([1, 5, 10, 15, 20], size=n_transactions)

# Tạo nhãn gian lận dựa trên các tiêu chí:
for i in range(n_transactions):
    amount = amounts[i]
    count_last_7_days = transaction_count_last_7_days[i]
    
    # Mặc định là không gian lận và không có lý do gian lận
    is_fraud = 0
    fraud_reason = ""

    # Kiểm tra số lượng giao dịch trong 7 ngày qua
    if count_last_7_days > 10:
        is_fraud = 1
        fraud_reason = "Số giao dịch trong 7 ngày quá nhiều."

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
    'time': pd.date_range(start='2024-01-01', periods=n_transactions, freq='H'),
    'amount': amounts[:n_transactions],
    'transaction_type': [
        'online', 'online', 'online', 'in-store', 'online',  
        'in-store', 'online', 'in-store', 'online', 'in-store',
        'online', 'online', 'in-store', 'online', 'in-store',
        'in-store', 'online', 'online', 'in-store', 'in-store',
        'online', 'online', 'in-store', 'online', 'in-store',
        'online', 'in-store', 'online', 'in-store', 'online'
    ][:n_transactions],
    'address': [fake.address().replace('\n', ', ') for _ in range(n_transactions)],
    'customer_id': [i for i in range(12345, 12345 + n_transactions)],
    'is_fraud': fraud_labels,
    'fraud_reason': fraud_reasons,
    'transaction_count_last_7_days': transaction_count_last_7_days,
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Kiểm tra lại dữ liệu
print(df.head())

# Xuất dữ liệu vào tệp CSV
df.to_csv('sample_transactions.csv', index=False)

# In thông báo sau khi tạo CSV thành công
print("Tệp CSV đã được tạo thành công với dữ liệu hợp lý!")