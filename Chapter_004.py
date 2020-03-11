# Chapter_004
# https://github.com/rickiepark/handson-ml/blob/master/03_classification.ipynb
# 업데이트날짜 : 20200312

# np.c_ : 컬럼추가
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 그림을 저장할 폴드
PROJECT_ROOT_DIR = "C:\Python\HandsOnML_code"
CHAPTER_ID = "model_train"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)




X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]   # 모든 샘플에 x0 = 1을 추가
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # 모든 샘플에 x0 = 1을 추가
# array([[1., 0.],
#        [1., 2.]])
y_predict = X_new_b.dot(theta_best)
y_predict

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15]) # x축, y축
save_fig("prediction_equation")
plt.show

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)


### 경사하강법
eta = 0.1   # 학습률
n_iterations = 1000 # 반복횟수
m = 100 # 샘플수

theta = np.random.randn(2, 1)   # 무작위초기화

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
