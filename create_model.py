import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import json
from sklearn.preprocessing import LabelEncoder
import joblib
from models import RandomForest
import ast

# Cấu hình Python để in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

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
    
    # Chọn ngày cố định là 1/12/2024
    base_date = pd.Timestamp('2024-12-02')  # Tất cả giao dịch của khách hàng sẽ ở ngày này
    
    for _ in range(count_last_1_days):
        amount = np.random.randint(50, 10000)
        current_time = base_date + pd.Timedelta(hours=np.random.randint(1, 24))  # Các giao dịch đều trong ngày 01/12/2024
        rolling_transaction_history[customer_id].append((amount, current_time))
        transaction_times[customer_id].append(current_time)

# Số tiền lớn liên tiếp (ngưỡng bất thường)
def is_large_transactions_consecutive(rolling_history, customer_id, threshold=3000, count=3):
    history = [amt for amt, _ in rolling_history.get(customer_id, [])]
    if len(history) < count:
        return False, ""
    
    # Kiểm tra 3 giao dịch liên tiếp vượt ngưỡng
    for i in range(len(history) - count + 1):
        if all(amount >= threshold for amount in history[i:i+count]):
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
    current_time = pd.Timestamp('2024-12-02') + pd.Timedelta(hours=i)  # Cố định ngày 01/12/2024
    count_last_1_days = len(transaction_times[customer_id]) + 1
    transaction_count_last_1_days.append(count_last_1_days)
    amount = np.random.randint(50, 10000)
    amounts.append(amount)
    is_abnormal = 0
    reason = ""

    if count_last_1_days > 10:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += "Số giao dịch trong 1 ngày quá nhiều."

    if amount > 7000:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += "Số tiền giao dịch lớn."

    # Các kiểm tra khác (giống như trước)
    large_consecutive, large_consecutive_reason = is_large_transactions_consecutive(
        rolling_transaction_history, customer_id
    )
    if large_consecutive:
        is_abnormal = 1
        reason += large_consecutive_reason

    night_abnormal, night_abnormal_reason = is_abnormal_at_night(
        rolling_transaction_history, customer_id, current_time
    )
    if night_abnormal:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += night_abnormal_reason

    small_consecutive, small_consecutive_reason = is_small_transactions_consecutive(
        rolling_transaction_history, customer_id
    )
    if small_consecutive:
        is_abnormal = 1
        if reason:
            reason += " "
        reason += small_consecutive_reason

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

    rolling_transaction_history[customer_id].append((amount, current_time))
    transaction_times[customer_id].append(current_time)
    rolling_transaction_history[customer_id] = sorted(rolling_transaction_history[customer_id], key=lambda x: x[1])
    labels.append(is_abnormal)
    reasons.append(reason)
    rolling_histories.append([(amt, t) for amt, t in rolling_transaction_history[customer_id]])
    transaction_counts.append(len(rolling_transaction_history[customer_id]))

# Tạo DataFrame
data = {
    'transaction_id': range(1, n_transactions + 1),
    'time': pd.date_range(start='2024-12-02', periods=n_transactions, freq='H'),
    'amount': amounts,
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
output_file = 'root_transactions.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Tệp CSV '{output_file}' đã được tạo thành công!")

# Chuyển đổi rolling_history thành các đặc trưng thống kê
def process_rolling_history(rolling_history, customer_id):
    history = [amt for amt, _ in rolling_history.get(customer_id, [])]
    if len(history) == 0:
        return 0, 0  # Trả về 0 nếu không có giao dịch
    mean = np.mean(history)
    std_dev = np.std(history)
    return mean, std_dev

# Thêm các đặc trưng thống kê vào DataFrame
df['rolling_history'] = df['rolling_history'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['rolling_history_mean'], df['rolling_history_std'] = zip(*df.apply(lambda row: process_rolling_history(rolling_transaction_history, row['customer_id']), axis=1))

# Huấn luyện mô hình dự đoán
X = df[['amount', 'customer_id', 'transaction_count_last_1_days', 'rolling_history_mean', 'rolling_history_std']].values
y = df['is_abnormal'].values

# Huấn luyện RandomForest
rf_model = RandomForest(n_trees=10, max_depth=10)
rf_model.fit(X, y)

# Lưu mô hình và label_encoder vào tệp
joblib.dump(rf_model, 'rf_model.pkl')
