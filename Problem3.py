import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score

# 加载数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 划分数据集为训练集和测试集，其中测试集占20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化K值和准确率列表
k_range = range(1, 31)
k_scores = []

# 进行交叉验证，选择最佳的K值
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# 找出准确率最高的K值
best_k = k_range[np.argmax(k_scores)]
print(f"Best K value: {best_k} with accuracy: {np.max(k_scores)}")

# 绘制K值与准确率的关系图
plt.plot(k_range, k_scores)
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('K-NN: Accuracy vs. K Value')
plt.show()

# 使用最佳的K值训练K-近邻分类器
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = knn_best.predict(X_test)

# 计算并打印测试集上的准确率
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with best K: {test_accuracy}")