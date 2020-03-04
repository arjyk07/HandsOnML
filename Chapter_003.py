# Chapter_003
# https://github.com/rickiepark/handson-ml/blob/master/03_classification.ipynb
# 업데이트날짜 : 20200304

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


(y_train_pred == (y_scores > 0)).all()
y_train_pred_90 = (y_scores > 150000)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("재현율", fontsize=16)
    plt.ylabel("정밀도", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
save_fig("precision_vs_recall_plot")
plt.show()



# p137 3.3.5 ROC 곡선
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('거짓 양성 비율')
    plt.ylabel('진짜 양성 비율')

plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)



from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3
                                    , method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # 양성 클래스에 대한 확률을 점수로 사용합니다.
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "랜덤 포레스트")
plt.legend(loc="lower right", fontsize=16)
# save_fig("roc_curve_comparison_plot")
plt.show()

roc_auc_score(y_train_5, y_scores_forest)
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest) # Out[54]: 0.9909682839739551
recall_score(y_train_5, y_train_pred_forest) # Out[55]: 0.8703191293119351




##### p141 3.4 다중분류
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])   # Out[56]: array(['4'], dtype='<U1')

some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores
# array([[-722944.99753139, -582352.47225096, -808404.40284928,
#         -279333.41716116,  -65089.41119025, -196765.20918831,
#         -574653.70029099, -194296.0539839 , -220146.77656639,
#         -188971.25882872]])

np.argmax(some_digit_scores)        # Out[58]: 4
sgd_clf.classes_
sgd_clf.classes_[4]


from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)
# 45 → 45개 분류기 생성함
# 이미지 하나를 분류하려면 45개 분류기를 모두 통과시켜 양성으로 분류된 클래스를 선택

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
# Out[62]: array(['9'], dtype=object)
# 랜덤 포레스트 분류기는 직접 샘플을 다중 클래스로 분류할 수 있기 때문에
# 별도로 사이킷런의 OvA나 OvO를 적용할 필요가 없음.

forest_clf.predict_proba([some_digit])
# Out[63]: array([[0.  , 0.01, 0.  , 0.  , 0.01, 0.02, 0.  , 0.01, 0.01, 0.94]])
# predict_proba() 메서드를 호출하면 분류기가 각 샘플에 부여한 확률을 알 수 있음.

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")





##### p144 3.5 에러분석
# 가능성이 높은 모델을 하나 찾았다.
# 성능 향상 시킬까? 에러 종류 분석

# 오차 행렬
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx     # 오차 행렬

plt.matshow(conf_mx, cmap=plt.cm.gray)
# 배열에서 가장 큰 값은 흰색, 가장 작은 값은 검은색으로 정규화되어 그려짐
save_fig("confusion_matrix_plot", tight_layout=False)
plt.show()


# 에러비율로 비교
# → 오차 행렬의 각 값을 대응되는 클래스의 이미지 개수로 나누어(절대 개수x) 에러비율 비교
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# 다른 항목은 그대로 유지하고 주대각선만 0으로 채워서 그래프 그리기
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
save_fig("confusion_matrix_errors_plot", tight_layout=False)
plt.show()




##### p148 3.6 다중 레이블 분류
from sklearn.neighbors import KNeighborsClassifier
y_train = y_train.astype(np.int64)

y_train_large = (y_train >= 7)          # 큰 값 여부(7,8,9)
y_train_odd = (y_train % 2 == 1)        # 홀수 여부
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
f1_score(y_multilabel, y_train_knn_pred, average="macro")



##### p150 3.7 다중 출력 분류
# 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화
# 즉, 값을 두 개 이상 가질 수 있다.
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test




def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
# 숫자 그림을 위한 추가 함수
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
save_fig("noisy_digit_example_plot")
plt.show()


knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)