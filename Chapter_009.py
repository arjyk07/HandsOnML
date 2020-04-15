# Chapter_009
# https://github.com/rickiepark/handson-ml/blob/master/09_up_and_running_with_tensorflow.ipynb
# https://tensorflow.blog/%EC%9C%88%EB%8F%84%EC%9A%B0%EC%A6%88%EC%97%90-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0/
# https://lsjsj92.tistory.com/257
# https://stackoverflow.com/questions/55290271/updating-anaconda-fails-environment-not-writable-error/57144988
# https://webnautes.tistory.com/1395


# 업데이트날짜 : 20200326, 20200411, 20200415
# 설치날짜 : 20200408

import tensorflow as tf
tf.__version__

# 공통
import numpy as np
# import os
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# 일관된 출력을 위해 유사난수 초기화
from tensorflow.python.framework import ops

# 데이터 로드
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]


# 일관된 출력을 위해 유사난수 초기화
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)



x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# with 블록 안에서는 with 문에서 선언한 세션이 기본 세션으로 지정
with tf.Session() as sess:
    x.initializer.run()     # tf.get_default_session().run(x.initializer)
    y.initializer.run()     # tf.get_default_session().run(y.initializer)
    result = f.eval()       # tf.get_default_session().run(f)

# global_variables_initializer() 함수
# 각 변수의 초기화를 일일이 실행하는 대신
# 초기화를 바로 실행하지 않고, 계산 그래프가 실행될 때 모든 변수를 초기화할 노드 생성
init = tf.global_variables_initializer() # init 노드 준비

with tf.Session() as sess:
    init.run()          # 실제 모든 변수를 초기화합니다.
    result = f.eval()



## 9.3 계산 그래프 관리
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
x2.graph is graph
x2.graph is tf.get_default_graph()


## 9.4 노드의 생애주기
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

# 한 노드를 평가할 때 텐서플로는
# 이 노드가 의존하고 있는 다른 노드들을 자동으로 찾아 먼저 평가한다.
with tf.Session() as sess:
    print(y.eval())         # 10
    print(z.eval())         # 15
# 이전에 평가된 w와 x를 재사용하지 않는다.

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)            # 10
    print(z_val)            # 15


## 9.5 텐서플로를 이용한 선형 회귀
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# theta정의 : theta = (XT * X)-1 * XT * y)

with tf.Session() as sess:
    theta_value = theta.eval()




## 9.6 경사 하강법 구현
### 9.6.1 직접 그래디언트 계산
# 경사하강법은 먼저 특성 벡터의 스케일을 조정해야 한다.
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="prediction")

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

with tf.Session() as sess:
    theta_value = theta.eval()

## 9.6.2 자동 미분 사용
gradients = tf.gradients(mse, [theta])[0]

training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("에포크", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("best_theta:")
print(best_theta)

# a, b에 대한 다음 함수의 편도함수 구하기
def my_func(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z

my_func(0.2, 0.3)


# 9.6.3 옵티마이저 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


# 9.7 훈련 알고리즘에 데이터 주입
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
print(B_val_2)

# 미니배치 경사 하강법 구현
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100
n_batches = int(np.ceil(m / batch_size))
n_epochs = 10

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

best_theta


# 9.8 모델의 저장과 복원
# 모델을 필요할 때 다시 쓸 수 있도록(다른 프로그램에서 사용하거나 다른 모델과 비교할 때 등)
# 모델 파라미터를 디스크에 저장해야 한다.
# 또한 훈련하는 동안 일정한 간격으로 체크포인트를 저장해두면
# 컴퓨터가 훈련 중간에 문제를 일으켜도 처음부터
# 다시 시작하지 않고 마지막 체크포인트부터 이어나갈 수 있다.
reset_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver() # 모델 저장

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("에포크", epoch, "MSE =", mse.eval())
            save_path = saver.save(sess, "/tmp/mymodel.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

best_theta

# 모델 복원
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval()

# Saver는 기본적으로 모든 변수를 각자의 이름으로 저장하고 복원
# theta를 "weights"와 같은 다른 이름으로 저장하고 복원하는
# Saver 객체를 원할 경우엔
saver = tf.train.Saver({"weights": theta})

# 기본적으로 Saver 객체는 .meta 확장자를 가진 두 번째 파일에
# 그래프 구조도 저장합니다. tf.train.import_meta_graph()함수를 사용하여
# 그래프 구조를 복원할 수 있습니다. 이 함수는 저장된 그래프를 기본 그래프로
# 로드하고 상태(즉, 변수값)를 복원할 수 있는 Saver 객체를 반환한다.
reset_graph()   # 빈 그래프로 시작
saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta") # 그래프 구조를 로드
theta = tf.get_default_graph().get_tensor_by_name("theta:0")

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  # 그래프 상태를 로드
    best_theta_restored = theta.eval()

np.allclose(best_theta, best_theta_restored)

# 이를 사용하면 그래프를 만든 파이썬 코드가 없이도 미리 훈련된 모델을 임포트할 수 있다.
# 모델을 저장하고 변경할 때도 매우 편리하다. 이전에 저장된 모델을 구축한 코드의
# 버전을 찾지 않아도 로드할 수 있다.
