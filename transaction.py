import pandas as pd
import numpy as np
from faker import Faker
from collections import defaultdict
import sys
import json
from sklearn.preprocessing import LabelEncoder
import joblib
from models import RandomForest

# Cấu hình Python để in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Khởi tạo Faker
fake = Faker()

# Đặt seed cho ngẫu nhiên để đảm bảo tái hiện kết quả
np.random.seed(42)

# Số lượng giao dịch cần tạo
n_transactions = 1000

# Lưu lịch sử giao dịch
rolling_transaction_history = defaultdict(list)
transaction_times = defaultdict(list)

# Hàm kiểm tra bất thường liên tục
def is_continuously_abnormal(rolling_history, customer_id, current_amount, threshold_count=3):
    history = [amt for amt, _ in rolling_history.get(customer_id, [])]  # Chỉ lấy số tiền
    if len(history) < 3:  # Yêu cầu ít nhất 3 giao dịch để xác định bất thường
        return False, ""
    
    # Tính ngưỡng bất thường
    mean = np.mean(history)
    std_dev = np.std(history)
    threshold = mean + 3 * std_dev  # Lớn hơn 3 lần độ lệch chuẩn

    # Kiểm tra nếu vượt ngưỡng nhiều lần
    abnormal_count = sum(1 for amount in history if amount > threshold)
    if abnormal_count >= threshold_count:
        return True, f"Số tiền giao dịch bất thường vượt thói quen {abnormal_count} lần."
    return False, ""

# Hàm kiểm tra giao dịch liên tục trong thời gian ngắn
def is_frequent_in_short_time(times, current_time, threshold_count=3, time_interval='1H'):
    times.append(current_time)
    time_series = pd.Series(times)
    time_series = pd.to_datetime(time_series)  # Đảm bảo time_series có kiểu dữ liệu datetime64
    time_series = time_series.sort_values()
    intervals = time_series.diff().dt.total_seconds().fillna(0)  # Khoảng thời gian giữa các giao dịch
    
    # Đếm số giao dịch trong thời gian ngắn (ví dụ: 1 giờ)
    short_time_count = sum(intervals <= pd.Timedelta(time_interval).total_seconds())
    if short_time_count > threshold_count:
        return True, f"Liên tục {short_time_count} giao dịch trong thời gian ngắn."
    return False, ""

# Danh sách chứa nhãn và lý do
labels = []
reasons = []
rolling_histories = []
transaction_counts = []
amounts = []  # Danh sách chứa số tiền giao dịch
transaction_count_last_1_days = []  # Danh sách chứa số giao dịch trong 1 ngày qua

# Tạo dữ liệu cho khách hàng
for customer_id in range(10000, 20000):
    count_last_1_days = np.random.choice([0, 4, 9, 14, 19])
    for _ in range(count_last_1_days):
        amount = np.random.randint(50, 10000)
        current_time = pd.Timestamp('2024-01-01') - pd.Timedelta(hours=np.random.randint(1, 24))
        rolling_transaction_history[customer_id].append((amount, current_time))
        transaction_times[customer_id].append(current_time)

# Số tiền lớn liên tiếp (ngưỡng bất thường)
def is_large_transactions_consecutive(rolling_history, customer_id, threshold=5000, count=3):
    history = [amt for amt, _ in rolling_history.get(customer_id, [])]
    if len(history) < count:
        return False, ""
    
    # Kiểm tra 3 giao dịch liên tiếp vượt ngưỡng
    for i in range(len(history) - count + 1):
        if all(amount > threshold for amount in history[i:i+count]):
            return True, f"3 giao dịch liên tiếp số tiền lớn hơn {threshold}."
    return False, ""

# Giao dịch bất thường vào ban khuya
def is_abnormal_at_night(rolling_history, customer_id, current_time, threshold=3000, night_hours=(23, 5)):
    hour = current_time.hour
    if hour >= night_hours[0] or hour < night_hours[1]:
        large_transactions = [
            amt for amt, t in rolling_history.get(customer_id, [])
            if amt > threshold and (t.hour >= night_hours[0] or t.hour < night_hours[1])
        ]
        if len(large_transactions) > 1:
            return True, f"Nhiều giao dịch lớn (> {threshold}) vào ban khuya."
    return False, ""

# Giao dịch liên tiếp dưới ngưỡng
def is_small_transactions_consecutive(rolling_history, customer_id, threshold=500, count=5):
    history = [amt for amt, _ in rolling_history.get(customer_id, [])]
    if len(history) < count:
        return False, ""
    
    # Kiểm tra 5 giao dịch liên tiếp dưới ngưỡng
    for i in range(len(history) - count + 1):
        if all(amount < threshold for amount in history[i:i+count]):
            return True, f"5 giao dịch liên tiếp số tiền nhỏ hơn {threshold}."
    return False, ""

# Thêm logic mới vào phần gán nhãn
for i in range(n_transactions):
    customer_id = np.random.randint(10000, 20000)
    current_time = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
    count_last_1_days = len(transaction_times[customer_id]) + 1
    transaction_count_last_1_days.append(count_last_1_days)
    amount = np.random.randint(50, 10000)
    amounts.append(amount)
    is_abnormal = 0
    reason = ""

    # Số tiền lớn liên tiếp
    large_consecutive, large_consecutive_reason = is_large_transactions_consecutive(
        rolling_transaction_history, customer_id
    )
    if large_consecutive:
        is_abnormal = 1
        reason += large_consecutive_reason

    # Giao dịch bất thường vào ban khuya
    night_abnormal, night_abnormal_reason = is_abnormal_at_night(
        rolling_transaction_history, customer_id, current_time
    )
    if night_abnormal:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += night_abnormal_reason

    # Giao dịch liên tiếp dưới ngưỡng
    small_consecutive, small_consecutive_reason = is_small_transactions_consecutive(
        rolling_transaction_history, customer_id
    )
    if small_consecutive:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += small_consecutive_reason

    # Các tiêu chí khác như trước
    abnormal, abnormal_reason = is_continuously_abnormal(rolling_transaction_history, customer_id, amount)
    if abnormal:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += abnormal_reason

    frequent, frequent_reason = is_frequent_in_short_time(
        transaction_times[customer_id], current_time, threshold_count=5, time_interval='30min'
    )
    if frequent:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += frequent_reason

    if count_last_1_days > 10:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += "Số giao dịch trong 1 ngày quá nhiều."

    if amount > 7000:  # Số tiền giao dịch quá lớn
        is_abnormal = 1
        if reason:
            reason += " "
        reason += "Số tiền giao dịch lớn."

    # Cập nhật lịch sử và sắp xếp theo thời gian
    rolling_transaction_history[customer_id].append((amount, current_time))
    transaction_times[customer_id].append(current_time)
    rolling_transaction_history[customer_id] = sorted(rolling_transaction_history[customer_id], key=lambda x: x[1])
    labels.append(is_abnormal)
    reasons.append(reason)
    rolling_histories.append([(amt, t) for amt, t in rolling_transaction_history[customer_id]])
    transaction_counts.append(len(rolling_transaction_history[customer_id]))

# Cân bằng dữ liệu (giảm nhãn bất thường nếu cần)
if sum(labels) > n_transactions // 2:
    idx_abnormal = [i for i, label in enumerate(labels) if label == 1]
    to_remove = len(idx_abnormal) - n_transactions // 2
    for idx in np.random.choice(idx_abnormal, to_remove, replace=False):
        labels[idx] = 0
        reasons[idx] = ""

# Tạo DataFrame
data = {
    'transaction_id': range(1, n_transactions + 1),
    'time': pd.date_range(start='2024-01-01', periods=n_transactions, freq='H'),
    'amount': amounts,
    'transaction_type': np.random.choice(['online', 'in-store'], size=n_transactions),
    'address': [fake.address().replace('\n', ', ') for _ in range(n_transactions)],
    'customer_id': np.random.randint(10000, 20000, size=n_transactions),
    'is_abnormal': labels,
    'reason': reasons,
    'transaction_count_last_1_days': transaction_count_last_1_days,
    'rolling_history': rolling_histories,
}

df = pd.DataFrame(data)

# Hiển thị dữ liệu mẫu
print("Dữ liệu mẫu:")
print(df.head())

# Chuyển đổi rolling_history từ list sang chuỗi JSON
df['rolling_history'] = df['rolling_history'].apply(lambda x: json.dumps(x, default=str))

# Lưu DataFrame vào tệp CSV
output_file = 'sample_transactions.csv'
df.to_csv(output_file, index=False)

print(f"\nDữ liệu đã được lưu vào {output_file}")