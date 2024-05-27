import pandas as pd
import numpy as np

# 直接从UCI机器学习库下载数据集  
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, delim_whitespace=True, names=column_names)

# 移除包含问号（?）的行，这些行表示数据缺失  
data = data[~data['horsepower'].str.contains('[?]')]
data['horsepower'] = pd.to_numeric(data['horsepower'])

# 选择特征和目标变量  
X = data[['cylinders', 'displacement', 'horsepower', 'weight']].values
y = data['mpg'].values


def linear_regression(X, y):
    # 在X矩阵的左侧添加一个全为1的列，用于表示偏置项
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # 使用正规方程求解参数
    theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best
j=0

print(X)
print(y)
theta_best = linear_regression(X, y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏置项
    return X_b.dot(theta)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


y_train_pred = predict(X_train, theta_best)
y_test_pred = predict(X_test, theta_best)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# 例如，只使用'displacement'和'horsepower'作为特征
X_reduced = data[['displacement', 'horsepower']].values
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

theta_best_reduced = linear_regression(X_train_reduced, y_train)
y_train_pred_reduced = predict(X_train_reduced, theta_best_reduced)
y_test_pred_reduced = predict(X_test_reduced, theta_best_reduced)

train_mse_reduced = mean_squared_error(y_train, y_train_pred_reduced)
test_mse_reduced = mean_squared_error(y_test, y_test_pred_reduced)

print(f"Train MSE with reduced features: {train_mse_reduced:.2f}")
print(f"Test MSE with reduced features: {test_mse_reduced:.2f}")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_best_scaled = linear_regression(X_train_scaled, y_train)
y_train_pred_scaled = predict(X_train_scaled, theta_best_scaled)
y_test_pred_scaled = predict(X_test_scaled, theta_best_scaled)

train_mse_scaled = mean_squared_error(y_train, y_train_pred_scaled)
test_mse_scaled = mean_squared_error(y_test, y_test_pred_scaled)

print(f"Train MSE with scaled features: {train_mse_scaled:.2f}")
print(f"Test MSE with scaled features: {test_mse_scaled:.2f}")