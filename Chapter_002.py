# Chapter_002
# https://github.com/rickiepark/handson-ml
# 업데이트날짜 : 20200228

# 독립적인 환경 만들기
# pip3 install --user --upgrade virtualenv
# virtualenv env

# 패키지 설치
# pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
# 모두 임포트
# python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"
# 주피터 실행
# jupyter notebook


# p79 2.3.2 데이터 다운로드
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