# Chapter_002
# https://github.com/rickiepark/handson-ml
# 추천강의 : https://programmers.co.kr/learn/courses/21
# 한글폰트 : # https://programmers.co.kr/learn/courses/21/lessons/950#
# 업데이트날짜 : 20200301

# 독립적인 환경 만들기
# pip3 install --user --upgrade virtualenv
# virtualenv env

# 패키지 설치
# pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
# 모두 임포트
# python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"
# 주피터 실행
# jupyter notebook

### p67
"""
    1. 큰 그림을 봅니다.
    2. 데이터를 구합니다.
    3. 데이터로부터 통찰을 얻기 위해 탐색하고 시각화합니다.
    4. 머신러닝 알고리즘을 위해 데이터를 준비합니다.
    5. 모델을 선택하고 훈련시킵니다.
    6. 모델을 상세하게 조정합니다.
    7. 솔루션을 제시합니다.
    8. 시스템을 론칭하고 모니터링하고 유지보수합니다.
"""



### p79 2.3.2 데이터 다운로드
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# fetch_housing_data()를 호출하면 작업공간에 datasets/housing 디렉터리를 만들고,
# housing.tgz 파일을 내려받고, 같은 디렉터리에 압축을 풀어 housing.csv 파일 생성
def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

# pandas를 사용하여 데이터 읽어들이기
# 위 함수는 모든 데이터를 담은 pandas의 데이터프레임 객체를 반환함
import pandas as pd
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()

housing.info()

# 각 카테고리마다 얼마나 많은 구역이 있는지 확인
housing["ocean_proximity"].value_counts()

housing.describe()

# 주피터 노트북의 매직명령
# matplotlib이 주피터 자체의 백엔드를 사용하도록 설정함
# 그러면 그래프는 노트북 안에 그려지게 됨
# %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()




### p85 2.3.4 테스트 세트 만들기
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

# 1) 매 번 다른 테스트 세트 생성할 수 있음 → 처음 실행에서 테스트 저장 그리고 다음번에 불러들이기
# 2) 매 번 다른 테스트 세트 생성할 수 있음 → 난수 초기값 지정 : ex) np.random.seed(42))
# 1), 2) 둘 다 업데이트된 데이터셋을 사용하려면 문제가 된다.

# 다른 일반적인 해결책 : 샘플의 식별자를 사용하여 테스트 세트로 보낼지 말지 정함
# ex) 각 샘플마다 식별자의 해시값을 계산하여,
# 해시의 마지막 바이트의 값이 51(256의 20% 정도)보다 작거나 같은 샘플만 테스트 세트로 보낼 수 있음.
# 이렇게 하면, 여러 번 반복 실행되면서 데이터셋이 갱신되더라도 동일한 테스트 세트 생성

# 무작위 샘플링
from zlib import crc32
import numpy as np

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # 'index' 열이 추가된 데이터프레임이 반환됨.

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split
# train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# random_state : 데이터 분할 시 셔플이 이루어지는데 이를 위한 시드값





# 계층적 샘플링_ocean_proximity 기반
# stratify : 지정한 data의 비율 유지. 예를 들어 Label set인 Y가 25%의 0과
# 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면,
# 나누어진 데이터셋도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.
housing["ocean_proximity"].value_counts()
train_set1, test_set1 = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing["ocean_proximity"])
train_set1["ocean_proximity"].value_counts() / len(train_set1)
test_set1["ocean_proximity"].value_counts() / len(test_set1)
"""
	        housing		    train_set		test_set	
cate	    count	ratio	count	ratio	count	ratio
OCEAN	    9,136 	44.26%	 7,309 	44.26%	 1,827 	44.26%
INLAND	    6,551 	31.74%	 5,241 	31.74%	 1,310 	31.73%
NEAR OCEAN	2,658 	12.88%	 2,126 	12.88%	   532 	12.89%
NEAR BAY	2,290 	11.09%	 1,832 	11.09%	   458 	11.09%
ISLAND	        5 	 0.02%	    4 	 0.02%	    1 	 0.02%
"""



# 전문가 의견 : 중간 소득이 중간 주택 가격을 예측하는 데 매우 중요해!
# 카테고리 수를 제한하기 위해 1.5로 나눔
import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# 5이상은 5로 구분
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)
housing["income_cat"].hist()
# np.ceil(a) : 각 원소값보다 크거나 같은 가장 작은 정수 값(천장값)으로 올림
a = np.array([-4.62, -2.19, 0, 1.57, 3.40, 4.06])
a
np.ceil(a)


# 계층적 샘플링_소득 카테고리 기반
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    train_set_strat = housing.loc[train_index]
    test_set_strat = housing.loc[test_index]
train_set_strat["income_cat"].value_counts() / len(train_set_strat)
test_set_strat["income_cat"].value_counts() / len(test_set_strat)

# 전체_소득 카테고리 기반
housing["income_cat"].value_counts() / len(housing)

# 무작위 샘플링_소득 카테고리 기반
from sklearn.model_selection import train_test_split
# train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)
train_set_sim, test_set_sim = train_test_split(housing, test_size=0.2, random_state=42)
train_set_sim["income_cat"].value_counts() / len(train_set_sim)
test_set_sim["income_cat"].value_counts() / len(test_set_sim)

# "income_cat" 컬럼 삭제
for set_ in (train_set_strat, test_set_strat):
    set_.drop("income_cat", axis=1, inplace=True)




########################################################
# p90 2.4 데이터 이해를 위한 탐색과 시각화
########################################################
housing = train_set_strat.copy()
### p91 2.4.1 지리적 데이터 시각화
housing.plot(kind="scatter", x="longitude", y="latitude")
# alpha 옵션으로 데이터 포인트 밀집된 영역 확인
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# 캘리포니아 주택 가격
# 코드 수정 : sharex=False 매개변수는 x-축의 값과 범례를 표시하지 못하는 버그를 수정합니다.
# 이는 임시 방편입니다(https://github.com/pandas-dev/pandas/issues/10611 참조).
# import matplotlib.pyplot as plt
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4
#              , s=housing["population"]/100, label="population", figsize=(10.7)
#              , c="median_house_value", cmap=plt.get_cmap("jet")
#              , colorbar=True, sharex=False)


# 한글출력 참고사이트1 : https://programmers.co.kr/learn/courses/21/lessons/950#
# 한글출력 참고사이트2 : https://m.blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221225208497&proxyReferer=https%3A%2F%2Fwww.google.com%2F
# 폰트 설정 방법 1
import matplotlib.pyplot as plt

# # 폰트 설정 방법 1
# plt.rc('font', family='NanumGothicOTF') # For MacOS
# plt.rc('font', family='NanumGothic') # For Windows
# print(plt.rcParams['font.family'])

# 폰트 설정 방법 2
# import matplotlib
# import matplotlib.font_manager as fm
# fm.get_fontconfig_fonts()
# # font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
# font_location = 'C:/Windows/Fonts/NanumGothic.ttf' # For Windows
# font_name = fm.FontProperties(fname=font_location).get_name()
# matplotlib.rc('font', family=font_name)
#
# ax = housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=housing["population"]/100, label="인구", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#     sharex=False)
# ax.set(xlabel='경도', ylabel='위도')
# plt.legend()


# p94 2.4.2 상관관계 조사
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income"
            , "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income"
            , y="median_house_value", alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)





########################################################
# p99 2.5 머신러닝 알고리즘을 위한 데이터 준비
########################################################
housing = train_set_strat.drop("median_house_value", axis=1)        # 훈련 세트를 위해 레이블 삭제
# 예측 변수와 타깃 값에 같은 변형을 적용하지 않기 위해 예측 변수와 레이블을 분리
housing_labels = train_set_strat["median_house_value"].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # 옵션 1 : 해당 구역을 제거
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # 옵션 2 : 전체 특성을 삭제
median = housing["total_bedrooms"].median()     # 옵션 3 : 어떤 값으로 채우기(0, 평균, 중간값 등)
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
sample_incomplete_rows["total_bedrooms"].head()


# 누락된 값을 중간값으로 대체
# sklearn.preprocessing.Imputer 클래스는 사이킷런 0.20 버전에서 사용 중지 경고가 발생하고
# 0.22 버전에서 삭제될 예정입니다. 대신 추가된 sklearn.impute.SimpleImputer 클래스를 사용합니다.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# 중간값이 수치형 특성에서만 계산될 수 있기 때문에 텍스트 특성인 ocean_proximity를 제외한 데이터 복사본을 생성합니다.
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

# imputer는 각 특성의 중간값을 계산해서 그 결과를 객체의 statistics_ 속성에 저장함.
# 새로운 데이터에서는 어떤 값이 누락될 지 모르기 때문에 모든 수치형 특성에 imputer를 적용하는 것이 바람직함.
imputer.statistics_
# 각 특성의 중간 값이 수동으로 계산한 것과 같은지 확인해 보세요
housing_num.median().values
# 이제 학습된 imputer 객체를 사용해 훈련 세트에서 누락된 값을 학습한 중간값으로 대체
X = imputer.transform(housing_num)

# numpy 배열에서 다시 pandas 데이터프레임으로 되돌리기
housing_tr = pd.DataFrame(X, columns=housing_num.columns
                            , index=list(housing.index.values))
housing_tr["total_bedrooms"].loc[sample_incomplete_rows.index.values]

# ocean_proximity는 텍스트라 그냥 남겨둠
housing_cat = housing["ocean_proximity"]
housing_cat.head()

# 대부분의 머신러닝 알고리즘은 숫자형을 다루므로
# 카테고리를 텍스트에서 숫자형으로 변환
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]
housing_categories

# one-hot encoding(원핫인코딩) : 한 특성만 1이고(핫), 나머지 특성은 0
from sklearn.preprocessing import OneHotEncoder
# 사이킷런 0.20 버전에서 OneHotEncoder의 동작 방식이 변경되었습니다.
# 종전에는 0~최댓값 사이의 정수를 카테고리로 인식했지만
# 앞으로는 정수나 문자열에 상관없이 고유한 값만을 카테고리로 인식합니다.
# 경고 메세지를 피하기 위해 categories 매개변수를 auto로 설정합니다.
encoder = OneHotEncoder(categories='auto')

# Scipy 희소행렬(sparse matrix)
# 수천 개의 카테고리가 있는 범주형 특성일 경우 매우 효율적
# 열이 수천 개인 행렬, 각 행은 1이 한 → 희소행렬은 0이 아닌 원소의 위치만 저장
# numpy의 reshape() 함수에서 -1은 차원을 지정하지 않는다는 뜻
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot

housing_cat_1hot.toarray()

# 텍스트 카테고리를 숫자 카테고리로,
# 숫자 카테고리를 원-핫 벡터로 바꿔주는
# 이 두 가지 변환을 Categorical Encoder를 사용하여 한꺼번에 처리 가능

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot

# 기본적으로 CategoricalEncoder는 희소행렬을 출력하지만
# 밀집행렬(dense matrix)을 원할 경우 encoding 매개변술르 "onehot-dense"로 지정가능
cat_encoder = CategoricalEncoder(encoding="onehot-dense")
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot
cat_encoder.categories_


# p106 나만의 변환기 생성
from sklearn.base import BaseEstimator, TransformerMixin

# 컬럼 인덱스
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# p108 2.5.5 변환 파이프라인
# 숫자 특성을 전처리하는 간단한 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

# DataFrameSelector는 나머지는 버리고 필요한 특성을 선택하여
# 데이터프레임을 넘파이 배열로 바꾸는 식으로 데이터를 변환합니다.
# 사이킷런이 DataFrame을 바로 사용하지 못하므로
# 수치형이나 범주형 컬럼을 선택하는 클래스를 만듭니다.
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# 하나의 큰 파이프라인에 이들을 모두 결합하여 수치형과 범주형 특성을 전처리합니다:
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

# 두 파이프라인을 하나로 합치기 : FeatureUnion
# from sklearn.pipeline import FeatureUnion
#
# full_pipeline = FeatureUnion(transformer_list = [
#     ("num_pipeline", num_pipeline),
#     ("cat_pipeline", cat_pipeline),
#     ])
from sklearn.compose import ColumnTransformer
full_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attribs),
        ("cat_encoder", OneHotEncoder(categories='auto'), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape



########################################################
# p111 2.6 모델 선택과 훈련
########################################################

# 선형회귀
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측: ", lin_reg.predict(some_data_prepared))
print("레이블: ", list(some_labels))

# RMSE 측정
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse        # Out[56]: 68380.0014239801 → 너무 심한 과소적합



# 결정 트리
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse       # Out[60]: 0.0 → 너무 심한 과대적합



# 교차 검증
# 사이킷런의 교차 검증 기능은 scoring 매개변수에 (낮을수록 좋은)
# 비용함수가 아니라 (클수록 좋은) 효용함수를 기대합니다.
# 그래서 평균 제곱 오차(MSE)의 반대값(즉, 음수값)을 계산하는
# neg_mean_squared_error 함수를 사용합니다.
# 이런 이유로 앞선 코드를 제곱근을 계산하기 전에 -scores로 부호 변경
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels
                         , scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
# Scores: [70869.46236857 68069.16577439 70147.99025059 69909.49321943
#  69942.60380611 74389.20488996 68937.69284924 72006.90026021
#  76753.12804918 69401.47064874]
# Mean: 71042.71121164162
# Standard deviation: 2530.400094036184
# 평균 71,042 +/- 2,530


# 비교 : 선형회귀모델 점수
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
# Scores: [66865.39177992 66602.13707868 70566.01451205 74177.10777808
#  67684.49019662 71103.16843468 64766.7449901  67707.69995726
#  71047.8078688  67684.6727271 ]
# Mean: 68820.52353233038
# Standard deviation: 2662.8368520268605
# 평균 68,820 +/- 2,662



# 랜덤포레스트
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10)
forest_reg.fit(housing_prepared, housing_labels)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels
                         , scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
forest_rmse_scores
display_scores(forest_rmse_scores)
# Scores: [51914.12916538 50554.8809285  52140.63654545 54175.59780076
#  52345.3454045  57113.38637516 51901.55279111 50297.03604654
#  55739.15103263 52518.66493645]
# Mean: 52870.03810264966
# Standard deviation: 2066.521470247491
# 평균 52,870 +/- 2,066







########################################################
# p115 2.7 모델 세부 튜닝
########################################################
# 2.7.1 그리드 탐색
from sklearn.model_selection import GridSearchCV

param_grid = [
    # 하이퍼파라미터 12(=3×4)개의 조합을 시도합니다.
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # bootstrap은 False로 하고 6(=2×3)개의 조합을 시도합니다.
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

forest_reg = RandomForestRegressor(n_estimators=10)
# 다섯 폴드에서 훈련하면 총 (12+6)*5=90번의 훈련이 일어납니다.
grid_search = GridSearchCV(forest_reg, param_grid, cv=5
                           , scoring='neg_mean_squared_error'
                           , return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# 최상의 파라미터 조합:
grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)




# 2.7.2 랜덤 탐색
# RandomizedSearchCV는 GridSearchCV와 거의 같은 방식으로 사용하지만 가능한 모든 조합을
# 시도하는 대신 각 반복마다 하이퍼파라미터에 임의의 수를 대입하여 지정한 횟수만큼 평가한다.


# 2.7.3 앙상블 방법
# 최상의 모델을 연결


# 2.7.4 최상의 모델과 오차 분석
# 최상의 모델을 분석하면 좋은 통찰을 얻는 경우가 많음
# 각 특성의 상대적인 중요도 확인
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

# 사이킷런 0.20 버전의 ColumnTransformer를 사용했기 때문에
# full_pipeline에서 cat_encoder를 가져옵니다. 즉 cat_pipeline을 사용하지 않았습니다:
# cat_one_hot_attribs = list(encoder.classes_)
# cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_encoder = full_pipeline.named_transformers_["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# 2.7.5 테스트 세트로 시스템 평가하기
# 테스트 세트에서 최종 모델을 평가
# 테스트 세트에서 예측 변수와 레이블을 얻은 후 full_pipeline을 사용해
# 데이터를 변환하고(fit_transform()이 아니라
# transform()을 호출해야 한다.
final_model = grid_search.best_estimator_

X_test = test_set_strat.drop("median_house_value", axis=1)
y_test = test_set_strat["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse      # Out[97]: 48170.67404204211
