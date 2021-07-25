# EM
最大期望算法（Expectation-maximization algorithm，又译为期望最大化算法），是在概率模型中寻找参数最大似然估计或者最大后验估计的算法，其中概率模型依赖于无法观测的隐性变量。

最大期望算法经过两个步骤交替进行计算：

- 第一步是计算期望（E），利用对隐藏变量的现有估计值，计算其最大似然估计值；
- 第二步是最大化（M），最大化在E步上求得的最大似然值来计算参数的值。M步上找到的参数估计值被用于下一个E步计算中，这个过程不断交替进行。
# GMM
高斯混合模型可以看作是由 K 个单高斯模型组合而成的模型，这 K 个子模型是混合模型的隐变量（Hidden variable）。一般来说，一个混合模型可以使用任何概率分布，这里使用高斯混合模型是因为高斯分布具备很好的数学性质以及良好的计算性能。
$$P(x|\theta) = \frac{1}{(2\pi)^{\frac{D}{2}}|\Sigma|^{\frac{1}{2}}}\exp{-\frac{{(x-\mu)}^T\Sigma (x-\mu)}{2}}$$

# PCA
PCA(主成分分析，Principal Component Analysis)是最常用的一种降维方法。PCA 的主要思想是将 D 维特征通过一组投影向量映射到 K 维上，这 K 维是全新的正交特征，称之为主成分，采用主成分作为数据的代表，有效地降低了数据维度，且保留了最多的信息。关于 PCA 的推导有两种方式：最大投影方差和最小投影距离。

- 最大投影方差：样本点在这个超平面上的投影尽可能分开
- 最小投影距离：样本点到这个超平面的距离都足够近

## 中心化

在开始 PCA 之前需要对数据进行预处理，即对数据中心化。设数据集 $X=\{x_1, x_2, ..., x_n\}$，其中 $x_i = \{x_{i1}, x_{i2}, ..., x_{id}\}$，即 $X$ 是一个 $n \times d$ 的矩阵。则此数据集的中心向量（均值向量）为：

$$
\mu = \frac 1 n \sum_{i=1}^n x_i
$$

对数据集每个样本均进行操作：$x_i = x_i - \mu$，就得到了中心化后的数据，此时有 $\displaystyle\sum_{i=1}^n x_i=0$。

中心化可以给后面的计算带来极大的便利，因为中心化之后的常规线性变换就是绕原点的旋转变化，也就是坐标变换。此时，协方差为 $S=\displaystyle\frac 1 n\sum_{i=1}^n x_i x_i^T=\frac 1 n X^T X$

设使用的投影坐标系的一组**标准正交基**为 $U_{k\times d}=\{u_1, u_2, ..., u_k\},\ k<d, u_i = \{u_{i1}, u_{i2}, ..., u_{id}\}$，故有 $UU^T=1$，使用这组基变换中心化矩阵 $X$，得降维压缩后的矩阵 $Y_{n \times k}=XU^T$，重建得到 $\hat X=YU=XU^TU$。

## 最大投影方差

对于任意一个样本 $x_i$，在新的坐标系中的投影为 $y_i=x_iU^T$，在新坐标系中的投影方差为 $y_i^Ty_i=Ux_i^T x_iU^T$。要使所有的样本的投影方差和最大，也就是求 $\displaystyle\arg\max_U \sum^n_{i=1} Ux_i^T x_iU^T$，即

$$
\arg \max_U\ tr(UX^TXU^T)\qquad s.t.\ UU^T=1
$$

求解：在 $u_1$ 方向投影后的方差

$$
\frac 1 n \sum_{i=1}^n\{u_1^Tx_i - u_1^T\mu\}^2 = \frac 1 n (Xu_1^T)^T(Xu_1^T)=\frac 1 n u_1X^TXu_1^T=u_1Su_1^T
$$

因为 $u_1$ 是投影方向，且已经假设它是单位向量，即 $u_1^Tu_1=1$，用拉格朗日乘子法最大化目标函数：

$$
L(u_1) = u_1^TSu_1 + \lambda_1(1-u_1^Tu_1)
$$

对 $u_1$ 求导，令导数等于 0，解得 $Su_1 = \lambda_1 u_1$，显然，$u_1$ 和 $\lambda_1$ 是一组对应的 $S$ 的特征向量和特征值，所以有 $u_1^TSu_1 = \lambda_1$，结合在 $u_1$ 方向投影后的方差式，可得求得最大化方差，等价于求最大的特征值。

要将 $d$ 维的数据降维到 $k$ 维，只需计算前 $k$ 个最大的特征值，将其对应的特征向量（$d\times 1$ 的）转为行向量（$1\times d$ 的）组合成特征向量矩阵 $U_{k\times d}$，则降维压缩后的矩阵为 $Y=XU^T$ 。

## 最小投影距离

现在考虑整个样本集，希望所有的样本到这个超平面的距离足够近，也就是得到 $Y$ 后，与 $X$ 的距离最小。即求：

$$
\begin{aligned}
\arg \min_U\sum^n_{i=1} || \hat x_i - x_i ||^2_2 &= \arg \min_U\sum\limits_{i=1}^n||x_iU^TU - x_i||_2^2\\
 & =\arg \min_U\sum\limits_{i=1}^n ((x_iU^TU)(x_iU^TU)^T - 2(x_iU^TU)x_i^T + x_ix_i^T)  \\
 & =\arg \min_U\sum\limits_{i=1}^n (x_iU^TUU^TUx_i^T - 2x_iU^TUx_i^T + x_ix_i^T) \\
 & =\arg \min_U \sum\limits_{i=1}^n (-x_iU^TUx_i^T + x_ix_i^T)  \\
 & =\arg \min_U -\sum\limits_{i=1}^n x_iU^TUx_i^T + \sum\limits_{i=1}^n x_ix_i^T \\
 & \Leftrightarrow \arg \min_U -\sum\limits_{i=1}^n x_iU^TUx_i^T \\
 & \Leftrightarrow \arg \max_U \sum\limits_{i=1}^n x_iU^TUx_i^T\\
 & = \arg \max_U\ tr(U(\sum\limits_{i=1}^n x_i^Tx_i)U^T) \\
 & =\arg \max_U\ tr(UX^TXU^T)\qquad s.t.\ UU^T=1
 \end{aligned}
$$

可以看到，这个式子与我们在最大投影方差中得到的式子是一致的，这就说明了这两种方式求得的结果是相同的。

> PCA 实现

```python
def pca(x, k):
    n = x.shape[0]
    mu = np.sum(x, axis=0) / n
    x_centralized = x - mu
    cov = (x_centralized.T @ x_centralized) / n
    values, vectors = np.linalg.eig(cov)
    index = np.argsort(values)  # 从小到大排序后的下标序列
    vectors = vectors[:, index[: -(k + 1) : -1]].T  # 把序列逆向排列然后取前k个，转为行向量
    return x_centralized, mu, vectors
```