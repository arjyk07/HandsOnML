# Chapter_009
# https://github.com/rickiepark/handson-ml/blob/master/08_dimensionality_reduction.ipynb
# https://tensorflow.blog/%EC%9C%88%EB%8F%84%EC%9A%B0%EC%A6%88%EC%97%90-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0/
# https://lsjsj92.tistory.com/257
# https://stackoverflow.com/questions/55290271/updating-anaconda-fails-environment-not-writable-error/57144988
# https://webnautes.tistory.com/1395


# 업데이트날짜 : 20200326
# 설치날짜 : 20200408

import tensorflow as tf
tf.__version__

# 공통
import numpy as np
# import os

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Masking, TimeDistributed
from tensorflow.keras.utils import plot_model

# 일관된 출력을 위해 유사난수 초기화
from tensorflow.python.framework import ops


def reset_graph(seed=42):
    # tf.reset_default_graph()
    ops.reset_default_graph()
    ops.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2


sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)


with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
result




init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
result


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

result