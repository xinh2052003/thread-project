from env import GridEnvironment
from agent import Robot
from vis import GridVisualizer
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


env = GridEnvironment(20, 50, 50)
# env.print_maze() #in mê cung lên terminal
robot = Robot(environment=env, alpha=0.1, gamma=0.9, epsilon=0.75)
robot.train(4000)

visual=GridVisualizer(env,robot)
visual.run_visualization() #in quá trình mê cung chạy
a = visual.way
b = a

# Đưa danh sách 'way' vào DataFrame
df_way = pd.DataFrame(a)
print(df_way)
# Các đặc trưng bạn muốn lấy
features = ['wall_top', 'wall_bot', 'wall_left', 'wall_right', 'bomb_top', 'bomb_bot', 'bomb_left', 'bomb_right', 'near_top', 'near_bot', 'near_left', 'near_right']
X_way = df_way[features] # Tập đặc trưng
y_way = df_way['action'] # Nhãn hành động





# ====== 1. Đọc dữ liệu ======
file_path = "dataset.csv"
df = pd.read_csv(file_path)

# ======= 2. Cắt đặc trưng môi trường từ 'wall_top' đến 'near_right' =======
all_columns = df.columns.tolist()
start_idx = all_columns.index("wall_top")
end_idx = all_columns.index("near_right")
# Lấy đặc trưng môi trường và nhãn hành động
X = df.iloc[:, start_idx:end_idx + 1]  # Tập đặc trưng môi trường
y = df["action"]  # Nhãn hành động

# ====== 3. Cây 1: Giải thích hành động thật sự của agent ======
clf_true = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_true.fit(X, y)



print("\n========== 🌳 LUẬT GIẢI THÍCH HÀNH ĐỘNG THẬT SỰ CỦA AGENT ==========\n")
print(export_text(clf_true, feature_names=features))
num_leaves = sum(clf_true.tree_.children_left == -1)
print("Số luật cây quyết định:", num_leaves)

# ====== 4. Các cây giải thích vì sao KHÔNG chọn từng hành động ======
actions = ["up", "down", "left", "right"]
clf_not_actions = {}

def build_not_action_tree(action):
    label = df["action"].apply(lambda a: a if a == action else f"not_{action}")
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, label)
    return clf

for action in actions:
    clf_not_actions[action] = build_not_action_tree(action)
    
# ====== Vẽ cây (tuỳ chọn) ======
def plot_tree_graph(clf, title):
    plt.figure(figsize=(20, 8))
    plot_tree(clf, feature_names=features, class_names=clf.classes_.astype(str), filled=True, rounded=True)
    plt.title(title)
    plt.show()
# plot_tree_graph(clf_true, "Cây quyết định")

# ====== Hàm giải thích chi tiết 1 trường hợp ======

def explain_case(clf, case_df, label_type, position=None):
    predicted = clf.predict(case_df)[0]
    node_indicator = clf.decision_path(case_df)
    decision_path = node_indicator.indices
    tree = clf.tree_
    features = clf.feature_names_in_

    i = 0
    for node in decision_path:
        
        # Nếu là nút lá, dừng
        if tree.children_left[node] == -1:
            break

        feature_index = tree.feature[node]
        
        feature_value = case_df.iloc[0, feature_index]

        #=================gán giải thích=====================
        if (feature_value == True): 
            dungsai = " "
        else:
            dungsai=" không "
        if feature_index in [0, 4]:
            tencot="phía trên"
        elif feature_index in [1, 5]:
            tencot="phía dưới"
        elif feature_index in [2,6]:
            tencot="bên trái"
        elif feature_index in [3, 7]:
            tencot="bên phải"
        elif feature_index == 8:
            tencot="agent lên trên"
        elif feature_index == 9:
            tencot="agent xuống dưới"
        elif feature_index == 10:
            tencot="agent qua trái"
        elif feature_index == 11:
            tencot="agent qua phải"
        else:
            tencot = ""
        if feature_index in [0, 1, 2, 3]:
            giatri = "có tường"
        elif feature_index in [4,5,6,7]:
            giatri = "có bomb"
        elif feature_index in [8,9,10,11]:
            giatri = "gần đích hơn"



        
        if label_type == "Agent":
            #======================In giải thích==================================
            print(f"{tencot}{dungsai}{giatri}", end=", ")

        elif label_type == "up":
            if feature_index % 4 ==0:
                print(f"{tencot}{dungsai}{giatri}", end=", ")

        elif label_type == "down":
            if feature_index % 4 ==1:
                print(f"{tencot}{dungsai}{giatri}", end=", ")
                
        elif label_type == "left":
            if feature_index % 4 ==2:
                print(f"{tencot}{dungsai}{giatri}", end=", ")
                
        elif label_type == "right":
            if feature_index % 4 ==3:
                print(f"{tencot}{dungsai}{giatri}", end=", ")
                
        i+=1



# ====== Phân tích đường đi ======
sample_indexes = [10, 20, 30]

for idx in sample_indexes:

    

    case_df = X.iloc[[idx]]
    
    true_action = y.iloc[idx]

    
    
    action = clf_true.predict(case_df)[0]

    if(action ==true_action):
        print(f"\n\n\n=================  Trường hợp [{df_way.iloc[idx]['x']}, {df_way.iloc[idx]['y']}] =================")
        print(f"Thực tế Agent '{true_action}'")
        print(f"🌳(Luật đầy đủ)Agent chọn {action} vì: ", end=" ")
        explain_case(clf_true, case_df, label_type="Agent")

        print(f"\n🌳Agent chọn '{action}' vì:", end=" ")
        explain_case(clf_not_actions[action], case_df, label_type=action)

        # Giải thích các lựa chọn không được chọn
        for act in actions:
            if str(act).strip().lower() != str(action).strip().lower():
                print(f"\n🚫Agent không chọn '{act}' vì:", end=" ")
                explain_case(clf_not_actions[act], case_df, label_type=act)

