import pandas as pd
import numpy as np
import streamlit as st
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Initialize Faker
fake = Faker()

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('sample_transactions.csv')

# Kiểm tra dữ liệu
st.title("Phát hiện gian lận giao dịch")
st.subheader("Dữ liệu giao dịch")
st.write(df)

# Huấn luyện mô hình
# Thêm cột lịch sử gian lận của khách hàng
df['fraud_history'] = df.groupby('customer_id')['is_fraud'].transform('max')
# Mã hóa transaction_type (nếu là phân loại)
df['transaction_type'] = df['transaction_type'].astype('category').cat.codes
# Cập nhật X với các đặc trưng mới
X = df[['amount', 'customer_id', 'transaction_type', 'transaction_count_last_7_days', 'fraud_history']]
y = df['is_fraud']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hàm để xây dựng một cây quyết định đơn giản
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Nếu chỉ có một lớp hoặc đạt độ sâu tối đa
        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return unique_classes[0]

        # Tìm điểm phân chia tốt nhất
        best_feature, best_threshold = self._best_split(X, y, n_features)
        if best_feature is None:
            return np.random.choice(unique_classes)

        # Tạo cây con
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        return parent_entropy - child_entropy

    def _entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log(probabilities + 1e-10))

    def predict(self, X):
        predictions = [self._predict(inputs, self.tree) for inputs in X]
        return np.array(predictions)

    def _predict(self, inputs, tree):
        if not isinstance(tree, tuple):
            return tree  # return the leaf node
        feature_index, threshold, left_tree, right_tree = tree
        if inputs[feature_index] < threshold:
            return self._predict(inputs, left_tree)
        else:
            return self._predict(inputs, right_tree)

# Hàm để tạo rừng cây (Forest)
class RandomForest:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = len(y)
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [np.bincount(tree_pred).argmax() for tree_pred in tree_preds.T]

# Huấn luyện mô hình Random Forest
rf_model = RandomForest(n_trees=20, max_depth=10)
rf_model.fit(X_train.values, y_train.values)

# Dự đoán trên tập kiểm tra
y_pred = rf_model.predict(X_test.values)

# Tính toán các chỉ số
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
roc_auc = roc_auc_score(y_test, y_pred)

# Hiển thị các kết quả
st.subheader("Ma trận nhầm lẫn")
st.write(conf_matrix)

st.subheader("Báo cáo phân loại")
st.write(class_report)

st.subheader("AUC-ROC")
st.write(roc_auc)

# Hiển thị kết quả
st.write("Kết quả phân loại:")
st.write(f"Độ chính xác: {class_report['accuracy']:.2f}")
st.write(f"Precision (cho gian lận): {class_report['1']['precision']:.2f}")
st.write(f"Recall (cho gian lận): {class_report['1']['recall']:.2f}")
st.write(f"F1-Score (cho gian lận): {class_report['1']['f1-score']:.2f}")

st.subheader("Kiểm tra ví dụ dự đoán")
amount = st.number_input("Nhập số tiền giao dịch:", min_value=1, max_value=100000, value=500)
customer_id = st.number_input("Nhập mã khách hàng:", min_value=1, max_value=100000, value=50)
transaction_type = st.selectbox("Chọn loại giao dịch:", ["online", "in-store"])

# Tìm số lượng giao dịch trong 7 ngày qua của khách hàng
if customer_id in df['customer_id'].values:
    transaction_count_last_7_days = df[df['customer_id'] == customer_id]['transaction_count_last_7_days'].max()
    fraud_history = df[df['customer_id'] == customer_id]['fraud_history'].max()
else:
    transaction_count_last_7_days = 0
    fraud_history = 0

# Dự đoán từ ví dụ
if st.button("Dự đoán gian lận"):
    # Kiểm tra nếu khách hàng đã bị gắn cờ gian lận trước đó
    if fraud_history == 1:
        st.write("Khách hàng này đã bị gắn cờ gian lận trước đó và không thể thực hiện thêm giao dịch.")
    else:
        # Kiểm tra các tiêu chí gian lận
        is_fraud = 0
        fraud_reason = ""

        if amount > 2000:
            is_fraud = 1
            fraud_reason += "Số tiền giao dịch lớn. "

        if transaction_count_last_7_days > 10:
            is_fraud = 1
            fraud_reason += "Số lượng giao dịch trong 7 ngày quá nhiều. "

        # Kiểm tra giao dịch liên tục trong khoảng thời gian ngắn
        last_transaction_time = df[df['customer_id'] == customer_id]['time'].max()
        if not pd.isnull(last_transaction_time):
            current_time = pd.Timestamp.now()
            if (current_time - pd.Timestamp(last_transaction_time)) < pd.Timedelta(minutes=5):
                is_fraud = 1
                fraud_reason += "Giao dịch liên tục trong khoảng thời gian ngắn. "

        # Cập nhật số lượng giao dịch trong 7 ngày qua
        transaction_count_last_7_days += 1

        # Cập nhật hoặc thêm giao dịch mới vào DataFrame
        if customer_id in df['customer_id'].values:
            # Cập nhật giao dịch hiện có
            df.loc[df['customer_id'] == customer_id, 'transaction_count_last_7_days'] = transaction_count_last_7_days
            df.loc[df['customer_id'] == customer_id, 'time'] = pd.Timestamp.now()
            df.loc[df['customer_id'] == customer_id, 'amount'] = amount
            df.loc[df['customer_id'] == customer_id, 'transaction_type'] = transaction_type
            df.loc[df['customer_id'] == customer_id, 'is_fraud'] = is_fraud
            df.loc[df['customer_id'] == customer_id, 'fraud_reason'] = fraud_reason.strip()
        else:
            # Thêm giao dịch mới
            new_transaction = {
                'transaction_id': df['transaction_id'].max() + 1,
                'time': pd.Timestamp.now(),
                'amount': amount,
                'transaction_type': transaction_type,
                'address': fake.address().replace('\n', ', '),
                'customer_id': customer_id,
                'is_fraud': is_fraud,
                'fraud_reason': fraud_reason.strip(),
                'transaction_count_last_7_days': transaction_count_last_7_days
            }
            df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)

        # Lưu DataFrame vào tệp CSV
        df.to_csv('sample_transactions.csv', index=False)

        # Hiển thị kết quả dự đoán và lý do gian lận
        result = "Gian lận" if is_fraud == 1 else "Không gian lận"
        st.write(f"Kết quả dự đoán: {result}")
        if is_fraud:
            st.write(f"Lý do gian lận: {fraud_reason.strip()}")

        # Hiển thị DataFrame cập nhật
        st.subheader("Dữ liệu giao dịch cập nhật")
        st.write(df)