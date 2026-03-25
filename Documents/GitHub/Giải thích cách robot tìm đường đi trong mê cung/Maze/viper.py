import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ====== 1. Đọc dữ liệu ======
file_path = "dataset.csv"
df = pd.read_csv(file_path)

# ====== 2. Lấy đặc trưng từ wall_top đến near_right ======
start_idx = df.columns.get_loc("wall_top")
end_idx = df.columns.get_loc("near_right")
features = df.columns[start_idx:end_idx + 1].tolist()
X = df[features]
y = df["action"]

# ====== 3. Cây 1: Giải thích hành động thật sự của agent ======
clf_true = DecisionTreeClassifier(max_depth=5, random_state=0)
clf_true.fit(X, y)

print("\n========== 🌳 LUẬT GIẢI THÍCH HÀNH ĐỘNG THẬT SỰ CỦA AGENT ==========\n")
print(export_text(clf_true, feature_names=features))

# ====== 4. Cây 2: Giải thích vì sao KHÔNG chọn một hành động cụ thể ======
target_action = "right"  # Hoặc "up", "down", "left" nếu muốn

# Gán nhãn 'not_right' nếu không phải hành động target, ngược lại giữ nguyên
df["not_target_action"] = df["action"].apply(lambda a: a if a == target_action else f"not_{target_action}")

y_not = df["not_target_action"]


clf_not = DecisionTreeClassifier(max_depth=5, random_state=0)
clf_not.fit(X, y_not)

print(f"\n========== 🚫 LUẬT GIẢI THÍCH VÌ SAO KHÔNG CHỌN '{target_action.upper()}' ==========\n")
print(export_text(clf_not, feature_names=features))

# ====== (Tuỳ chọn) Vẽ cây cho trực quan nếu cần ======
def plot_tree_graph(clf, title):
    plt.figure(figsize=(20, 8))
    plot_tree(clf, feature_names=features, class_names=clf.classes_.astype(str), filled=True, rounded=True)
    plt.title(title)
    plt.show()

# plot_tree_graph(clf_true, "Cây hành động thật sự")
# plot_tree_graph(clf_not, f"Cây loại trừ hành động '{target_action}'")

def explain_case(clf, case_df, label_type, position=None):
    predicted = clf.predict(case_df)[0]
    node_indicator = clf.decision_path(case_df)
    tree = clf.tree_
    features = clf.feature_names_in_

    print(f"\n🎯 Vị trí: {position}")
    print(f"👉 {label_type} dự đoán: {predicted}")
    print("🧠 Luồng logic:")

    for i, node_id in enumerate(node_indicator.indices):
        if tree.children_left[node_id] == -1:
            print(f"{'   '*i}--> Đến lá, kết luận: {predicted}")
            break

        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = features[feature_index]
        feature_value = case_df.iloc[0, feature_index]

        decision = "<=" if feature_value <= threshold else ">"
        print(f"{'   '*i}{feature_name} ({feature_value}) {decision} {threshold:.2f}")

# ====== Ví dụ ngẫu nhiên ======
sample_indexes = [10, 20, 30]  # bạn có thể thay đổi chỉ số
target_action = "right"  # hành động bạn muốn phân tích vì sao KHÔNG được chọn

for idx in sample_indexes:
    print(f"\n================= 🧩 Ví dụ {idx} =================")
    case_df = X.iloc[[idx]]
    true_action = y.iloc[idx]

    print("\n🌳 Cây 1 - Agent thật chọn hành động gì và vì sao:")
    explain_case(clf_true, case_df, label_type="Agent")

    print(f"\n🚫 Cây 2 - Vì sao agent không chọn hành động '{target_action}':")
    explain_case(clf_not, case_df, label_type=f"Loại '{target_action}'")
