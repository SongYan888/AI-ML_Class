from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分数据集为训练集和测试集，其中测试集占20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化决策树分类器，默认参数
clf = DecisionTreeClassifier(random_state=42)
# 训练模型
clf.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = clf.predict(X_test)

# 计算并打印性能指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')  # 或者选择其他平均方式，如'micro', 'weighted'等
f1 = f1_score(y_test, y_pred, average='macro')  # 同样可以选择其他平均方式

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 尝试不同的参数设置
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)
recall1 = recall_score(y_test, y_pred, average='macro')  # 或者选择其他平均方式，如'micro', 'weighted'等
f11 = f1_score(y_test, y_pred, average='macro')  # 同样可以选择其他平均方式

print(f"Accuracy: {accuracy1}")
print(f"Recall: {recall1}")
print(f"F1 Score: {f11}")
# 重新计算并打印性能指标...

