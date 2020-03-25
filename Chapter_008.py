# Chapter_008
# https://github.com/rickiepark/handson-ml/blob/master/08_dimensionality_reduction.ipynb
# 업데이트날짜 : 20200325

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_swiss_roll
from matplotlib import gridspec

from six.moves import urllib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding


np.random.seed(42)

PROJECT_ROOT_DIR = "C:\Python\HandsOnML_code"
CHAPTER_ID = "dim_reduction"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# 3D 데이터셋 생성
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
# ● np.zeros(), np.ones(), np.empty() 함수는
# 괄호 안에 쓴 숫자 개수만큼의 '0', '1', '비어있는 배열' 공간 생성

X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
# ● np.random.randn(n, m) - 표준 정규 분포에서 무작위 배열 생성
# ● np.random.rand(n, m, ...) - 다차원 무작위 배열 생성
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

""" ###################################### 
    투영 
###################################### """
# 1. SVD분해를 사용한 PCA
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

m, n = X.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

np.allclose(X_centered, U.dot(S).dot(Vt))

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

X2D_using_svd = X2D

# 2.사이킷런을 사용한 PCA
pca = PCA(n_components=2)

X2D = pca.fit_transform(X)
X2D[:5]
X2D_using_svd[:5] # SVD 방식을 사용한 것과 통일한 투영 결과


### 복원
# 평면(PCA 2D 부분공간)에 투영된 3D 포인트를 복원한다.
X3D_inv = pca.inverse_transform(X2D)

# 투영 단계에서 일부 정보를 잃어버리기 때문에 복원된 3D 포인트가
# 원본 3D 포인트와 완전히 똑같지는 않다.
np.allclose(X3D_inv, X)

# 재구성 오차 산출
np.mean(np.sum(np.square(X3D_inv - X), axis=1))

# SVD 방식의 역변환
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])

# 사이킷런의 PCA 클래스는 자동으로 평균을 뺏던 것을 복원해주기 때문에
# 두 방식의 재구성 오차가 동일하지는 않다.
# 하지만 평균을 빼면 동일한 재구성을 얻을 수 있다.
np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_)



# 주성분 확인(PCA 객체 사용)
pca.components_
# array([[-0.95060684, -0.26754816, -0.15736776],
#        [ 0.29982336, -0.92266892, -0.24246242]])
pca.components_.T[:,0]
# 첫 번째 주성분 : array([-0.95060684, -0.26754816, -0.15736776])


# 주성분 확인(SVD 방법)
Vt[:2]
# array([[-0.95060684, -0.26754816, -0.15736776],
#        [ 0.29982336, -0.92266892, -0.24246242]])


# 설명된 분산 비율 확인
pca.explained_variance_ratio_
# array([0.85752038, 0.13332972])
# → 첫번째 차원이 85.8% 분산 포함, 두번째 차원 13.3%의 분산 설명
# → 2D로 투영했기 때문에 분산의 0.9%의 분산을 잃었다.



# SVD 방식을 사용했을 때 설명된 분산의 비율 설명 방법(s는 행렬 S의 대각성분)
np.square(s) / np.square(s).sum()


###### 그래프(p.272, 2차원에 가깝게 배치된 3차원 데이터셋)
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
# np.linspace(start, stop, num)
# - 배열 시작값, 배열의 끝값, start~stop 사이의 몇 개의 일정한 간격
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)
# meshgrid 명령 등을 통해 행단위와 열단위로 각각 배열을 정방(square) 행렬 선언


# 평면 표현
C = pca.components_
R = C.T.dot(C) # (3,2) * (2,3)
# array([[ 0.99354742, -0.02230458,  0.07689898],
#        [-0.02230458,  0.92289995,  0.26581599],
#        [ 0.07689898,  0.26581599,  0.08355264]])
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')

X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

# 주성분 축
ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
np.linalg.norm(C, axis=0) # array([0.99676849, 0.96067682, 0.28905473])
ax.add_artist(Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.add_artist(Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.plot([0], [0], [0], "k.")

# 투영
for i in range(m):
    if X[i, 2] > X3D_inv[i, 2]:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
    else:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")

ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")

ax.set_xlabel("$x_1$", fontsize=18, labelpad=7)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=7)
ax.set_zlabel("$x_3$", fontsize=18, labelpad=4)

ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("dataset_3d_plot")
plt.show()


###### 그래프(p.273, 투영하여 만들어진 새로운 2D 데이터셋)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

ax.plot(X2D[:, 0], X2D[:, 1], "k+")
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
ax.plot([0], [0], "ko")
ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.set_xlabel("$z_1$", fontsize=18)
ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
ax.axis([-1.5, 1.3, -1.2, 1.2])     # 축 조정
ax.grid(True)       # 그래프 내 grid효과
save_fig("dataset_2d_plot")




""" ###################################### 
    매니폴드 학습
###################################### """
##### 그래프(p.274, 스위스롤 데이터셋)
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
# 라벨
ax.set_xlabel("$x_1$", fontsize=18, labelpad=7)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=7)
ax.set_zlabel("$x_3$", fontsize=18)
# 축 설정
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("swiss_roll_plot")
plt.show()


##### 그래프(p.274, 스위스롤 펼쳐놓기)
# (p.274, 평면에 그냥 투영시켜서 뭉개진 것(왼쪽)과 스위스롤을 펼쳐놓은 것(오른쪽)
plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis(axes[:4])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0, labelpad=10)
plt.grid(True)

plt.subplot(122)
plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)

save_fig("squished_swiss_roll_plot")
plt.show()


##### 그래프(p.276, 저차원에서 항상 간단하지 않은 결정 경계
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
axes = [-11.5, 14, -2, 23, -12, 15]

x2s = np.linspace(axes[2], axes[3], 10)
x3s = np.linspace(axes[4], axes[5], 10)
x2, x3 = np.meshgrid(x2s, x3s)

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = X[:, 0] > 5
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot_wireframe(5, x2, x3, alpha=0.5)
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18, labelpad=7)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=7)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("manifold_decision_boundary_plot1")
plt.show()




fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0, labelpad=7)
plt.grid(True)

save_fig("manifold_decision_boundary_plot2")
plt.show()



fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = 2 * (t[:] - 4) > X[:, 1]
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18, labelpad=7)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=7)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("manifold_decision_boundary_plot3")
plt.show()




fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.plot([4, 15], [0, 22], "b-", linewidth=2)

plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0, labelpad=7)
plt.grid(True)

save_fig("manifold_decision_boundary_plot4")
plt.show()




##### PCA
##### 그래프(p.277, 투영할 부분 공간 선택하기)
angle = np.pi / 5
stretch = 5
m = 200

np.random.seed(3)
X = np.random.randn(m, 2) / 10  #(100, 2)
X = X.dot(np.array([[stretch, 0],[0, 1]])) # (100,2) * (2,2) → stretch
# np.array([[stretch, 0], [0, 1]])
# array([[5, 0],
#        [0, 1]])
X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) # rotate
# ([0.8090169943749475, 0.5877852522924731],
#  [-0.5877852522924731, 0.8090169943749475])


# 주성분 축?
u1 = np.array([np.cos(angle), np.sin(angle)])   # array([0.80901699, 0.58778525])
u2 = np.array([np.cos(angle - 2 * np.pi/6), np.sin(angle - 2 * np.pi/6)])   # array([ 0.91354546, -0.40673664])
u3 = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])   # array([ 0.58778525, -0.80901699])

X_proj1 = X.dot(u1.reshape(-1, 1))
X_proj2 = X.dot(u2.reshape(-1, 1))
X_proj3 = X.dot(u3.reshape(-1, 1))


plt.figure(figsize=(8,4))
plt.subplot2grid((3,2), (0, 0), rowspan=3)
plt.plot([-1.4, 1.4], [-1.4*u1[1]/u1[0], 1.4*u1[1]/u1[0]], "k-", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4*u2[1]/u2[0], 1.4*u2[1]/u2[0]], "k--", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4*u3[1]/u3[0], 1.4*u3[1]/u3[0]], "k:", linewidth=2)

plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
plt.axis([-1.4, 1.4, -1.4, 1.4])

plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')

plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)

plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)

plt.grid(True)


plt.subplot2grid((3,2), (0, 1))
plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3,2), (1, 1))
plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3,2), (2, 1))
plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.axis([-2, 2, -1, 1])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)

save_fig("pca_best_projection")
plt.show()





##### MNIST
##### 그래프(P.283, 분산의 95%가 유지된 MNIST 압축)
mnist = fetch_openml('mnist_784', version=1)

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)


##### 점진적 PCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="") # not shown in the book
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)



##### 커널 PCA
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)




### LLE
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)

plt.title("LLE를 사용하여 펼쳐진 스위스 롤", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)

save_fig("lle_unrolling_plot")
plt.show()



# ● reshape : 기존 데이터는 유지하고 차원과 형상을 바꾸는 데 사용
# https://m.blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221415471793&proxyReferer=https%3A%2F%2Fwww.google.com%2F
# 파라미터로 입력한 차원에 맞게 변경. -1로 지정하면 나머지는 자동으로 맞춘다
X = np.linspace(1.0, 100.0, num=100)
X.shape

# 2차원 4*25 변환
X_4_25 = X.reshape(4, 25, order='C') # C : 뒤 차원부터 변경하고 앞 쪽 차원을 변경
# 3차원 4*5*5 변환
X_4_5_5 = X_4_25.reshape(4, 5, 5, order='C')
# 2차원 4*25 변환
X_4_25_prime = X_4_5_5.reshape(4, 25, order='C')
# 3차원 4*5*5 변환
X_4_5_5_prime = X_4_25_prime.reshape(4, 5, 5, order='C')
# 1차원 100 변환
X_prime = X_4_5_5_prime.reshape(100, order='C')
