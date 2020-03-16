# Chapter_006
# https://github.com/rickiepark/handson-ml/blob/master/06_decision_trees.ipynb
# 업데이트날짜 : 20200317

import numpy as np
import os

# 일관된 출력을 위해 유사난수 초기화

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
# 참고 : https://stackoverflow.com/questions/33433274/anaconda-graphviz-cant-import-after-installation
# The key to understanding is that conda install graphviz
# does not do the same thing as pip install graphviz.
# conda install python-graphviz does. conda install graphviz installs the binaries,
# which is the same as downloading and installing GraphViz from their website.

# 그림을 저장할 폴드
PROJECT_ROOT_DIR = "C:\Python\HandsOnML_code"
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)



iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)



# 시각화
export_graphviz(tree_clf
                , out_file=image_path("iris_tree.dot")
                , feature_names=iris.feature_names[2:]
                , class_names=iris.target_names
                , rounded=True
                , filled=True)

with open("images/decision_trees/iris_tree.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='iris_tree', directory='images/decision_trees/')


# 클래스와 클래스 확률 예측
tree_clf.predict_proba([[5, 1.5]])  # array([[0.        , 0.90740741, 0.09259259]])
tree_clf.predict([[5, 1.5]])    # array([1])


