# Chapter_003
# https://github.com/rickiepark/handson-ml/blob/master/03_classification.ipynb
# 업데이트날짜 : 20200303

# 파이썬 2와 파이썬 3 지원
from __future__ import division, print_function, unicode_literals

# 공통
import numpy as np
import os
import matplotlib.pyplot as plt


# 맷플롯립 설정
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
matplotlib.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

# 일관된 출력을 위해 유사난수 초기화
np.random.seed(42)

# 그림을 저장할 폴드
PROJECT_ROOT_DIR = "C:\Python\HandsOnML_code"
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)






# MNIST 데이터셋 불러오기
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist

X, y = mnist["data"], mnist["target"]
X.shape
y.shape

# 모두 임포트
# python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"
# 주피터 실행
# jupyter notebook

# 이미지가 28*28 픽셀, 0(흰색) ~ 255(검은색)까지의 픽셀 강도
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)       # 배열 28*28로 크기를 바꿈

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary
           , interpolation="nearest")
plt.axis("off")
plt.show()

y[36000]

# 행 번호로 train_set, test_set 분리
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# train_set, test_set shuffle
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# p126 3.2 이진분류기 훈련
# 5 분류기
y_train_5 = (y_train == '5')      # 9는 True고, 다른 숫자는 모두 False
y_test_5 = (y_test == '5')


# 분류 모델 적용 : sklearn의 SGDClassifier(확률적 경사 하강법, Stochastic Gradient Descent)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)
"""
SGDClassifier(alpha=0.0001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=5,
              n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
              random_state=42, shuffle=True, tol=0.001, validation_fraction=0.1,
              verbose=0, warm_start=False)
"""
# 테스트
sgd_clf.predict([some_digit])       # array([False]) → 오답!

# p127 3.3 성능 측정
# 교차 검증
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# array([0.83935, 0.85125, 0.8676 ])

# 더미분류기 - 모든 이미지를 '5 아님' 클래스로 분류하는 더미분류기 생성
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.909  , 0.90745, 0.9125 ]) - 이미지의 10% 정도만 숫자 5이기 때문에
# 무조건 '5 아님'으로 예측하면, 정확히 맞출 확률이 90% 이상
"""
    ex) X           : 1~256, 1~256, 1~256, 1~256, 1~256, 1~256, ...
    ex) y_train     : 1,2,3,4,5,0
        Never0Classifier() → 5을 제외하고, 전부 다 False로 바꿈
    ex) y_train_0   : False, False, False, False, True, False
"""




# p129 3.3.2 오차행렬
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
#                     예측 5x    예측 5o
#   실제 5x    array([[54058  ,   521],
#   실제 5o            [ 1899 ,  3522]], dtype=int64)


# p132 3.3.3 정밀도와 재현율
from sklearn.metrics import precision_score, recall_score
# 정밀도 = TP / (TP + FP)
precision_score(y_train_5, y_train_pred)        # 0.8711352955725946 = 3522 / (3522 + 521)
# 재현율 = TP / (TP + FN)
recall_score(y_train_5, y_train_pred)           # 0.6496956281128943 = 3522 / (3522 + 1899)
# F1 점수 = 2 * (정밀도 * 재현율) / (정밀도 + 재현율)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)               # 0.7442941673710904



# p133 3.3.4 정밀도/재현율 트레이드오프
# SGDClassifier가 분류를 어떻게 하는가?
# 이 분류기는 결정함수(decision function)를 사용하여 각 샘플의 점수를 계산함
# 이 점수가 임계값보다 크면 샘플을 양성(여기서는 5), 그렇지 않으면 음성(여기서는 5x) 클래스에 할당함
y_scores = sgd_clf.decision_function([some_digit])
y_scores            # array([-400042.39513131])     - 예측에 사용한 점수
threshold = -500000       # 임계값
y_some_digit_pred = (y_scores > threshold)      # array([True]) - 임계값이 낮아서 5를 감지함

threshold2 = 0
y_some_digit_pred2 = (y_scores > threshold2)    # array([False]) - 임계값을 높이니 5를 감지못함(재현율↓)


# 적절한 임계값은 어떻게 정하는가?
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3
                             , method="decision_function")
# 모든 샘플의 점수 구함
# array([-545086.1906455 , -200238.20632717, -366873.76172794, ...,
#        -626454.84454281, -716313.74931348, -581950.04601147])

# 모든 임계값에 대해 정밀도와 재현율 산출
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="정밀도")
    plt.plot(thresholds, recalls[:-1], "g-", label="재현율")
    plt.xlabel("임곗값")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
save_fig("precision_recall_vs_threshold_plot")
plt.show()


