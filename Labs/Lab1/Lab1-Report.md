<br/>
<br/>
<center> <font size = 5> 哈尔滨工业大学计算学部 </font></center>
<br/>
<br/>
<center> <font size = 15> 实验报告 </font></center>
<br/>
<br/>
<br/>
<center> <font size = 5> 
《机器学习》 <br/>
实验一：多项式拟合正弦函数 
</font></center>
<br/>
<br/>
<center> <font size = 4> 学号：1183710109 </font></center>
<center> <font size = 4> 姓名：郭茁宁 </font></center>

<div STYLE="page-break-after: always;"></div>

# 一、实验目的

掌握最小二乘法求解（无惩罚项的损失函数），掌握增加惩罚项（2 范数）的损失函数优化，梯度下降法、共轭梯度法，理解过拟合、客服过拟合的方法（如增加惩罚项、增加样本）。

# 二、实验要求及实验环境

## 实验要求

1. 生成数据，加入噪声；
2. 用高阶多项式函数拟合曲线；
3. 用解析解求解两种 loss 的最优解（无正则项和有正则项）
4. 优化方法求解最优解（梯度下降，共轭梯度）；
5. 用你得到的实验数据，解释过拟合。
6. 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。
7. 语言不限，可以用 matlab，python。求解解析解时可以利用现成的矩阵求逆。梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例如 pytorch，tensorflow 的自动微分工具。

## 实验环境

- OS: Win 10
- Python 3.6.8

# 三、算法原理和设计

## 1、生成数据算法

主要是利用$sin(2\pi x)$函数产生样本，其中$x$均匀分布在$[0, 1]$之间，对于每一个目标值$t=sin(2\pi x)$增加一个$0$均值，方差为$0.25$的高斯噪声。

```python
def get_data(
    x_range: (float, float) = (0, 1),
    sample_num: int = 10,
    base_func=lambda x: np.sin(2 * np.pi * x),
    noise_scale=0.25,
) -> "pd.DataFrame":
    X = np.linspace(x_range[0], x_range[1], num=sample_num)
    Y = base_func(X) + np.random.normal(scale=noise_scale, size=X.shape)
    data = pd.DataFrame(data=np.dstack((X, Y))[0], columns=["X", "Y"])
    return data
```

## 2、利用高阶多项式函数拟合曲线(不带惩罚项)

利用训练集合，对于每个新的$\hat{x}$，预测目标值$\hat{t}$。采用多项式函数进行学习，即利用式$(1)$来确定参数$w$，假设阶数$m$已知。

$$
y(x, w) = w_0 + w_1x + \dots + w_mx^m = \sum_{i = 0}^{m}w_ix^i \tag{1}
$$

采用最小二乘法，即建立误差函数来测量每个样本点目标值$t$与预测函数$y(x, w)$之间的误差，误差函数即式$(2)$

$$
E(\bold{w}) = \frac{1}{2} \sum_{i = 1}^{N} \{y(x_i, \bold{w}) - t_i\}^2 \tag{2}
$$

将上式写成矩阵形式如式$(3)$

$$
E(\bold{w}) = \frac{1}{2} (\bold{Xw} - \bold{T})'(\bold{Xw} - \bold{T})\tag{3}
$$

其中$\bold{X} =
\left[
\begin{matrix}
 1      & x_1      & \cdots & x_1^m      \\
 1      & x_2      & \cdots & x_2^m      \\
 \vdots & \vdots & \ddots & \vdots \\
 1      & x_N      & \cdots & x_N^m      \\
\end{matrix}
\right], 
\bold{w} = 
\left[
\begin{matrix}
    w_0 \\ w_1 \\ \vdots \\ w_m
\end{matrix}
\right],
\bold{T} = 
\left[
\begin{matrix}
   t_1 \\ t_2 \\ \vdots \\ t_N 
\end{matrix}
\right]$
通过将上式求导我们可以得到式$(4)$

$$
\cfrac{\partial E}{\partial \bold{w}} = \bold{X'Xw} - \bold{X'T} \tag{4}
$$

令 $\cfrac{\partial E}{\partial \bold{w}}=0$ 我们有式$(5)$即为$\bold{w^*}$

$$
\bold{w^*} = (\bold{X'X})^{-1}\bold{X'T} \tag{5}
$$

由$(5)$实现

```python
def get_params(x_matrix, t_vec) -> [float]:
    return np.linalg.pinv(x_matrix.T @ x_matrix) @ x_matrix.T @ t_vec
```

## 3、带惩罚项的多项式函数拟合曲线

在不带惩罚项的多项式拟合曲线时，在参数多时$\bold{w}^*$具有较大的绝对值，本质就是发生了过拟合。对于这种过拟合，我们可以通过在优化目标函数式$(3)$中增加$\bold{w}$的惩罚项，因此我们得到了式$(6)$。

$$
\widetilde{E}(\bold{w}) = \frac{1}{2} \sum_{i=1}^{N} \{y(x_i, \bold{w}) - t_i\}^2 + \cfrac{\lambda}{2}|| \bold{w} || ^ 2 \tag{6}
$$

同样我们可以将式$(6)$写成矩阵形式， 我们得到式$(7)$

$$
\widetilde{E}(\bold{w}) = \frac{1}{2}[(\bold{Xw} - \bold{T})'(\bold{Xw} - \bold{T}) + \lambda \bold{w'w}]\tag{7}
$$

对式$(7)$求导我们得到式$(8)$

$$
\cfrac{\partial \widetilde{E}}{\partial \bold{w}} = \bold{X'Xw} - \bold{X'T} + \lambda\bold{w} \tag{8}
$$

令 $\cfrac{\partial \widetilde{E}}{\partial \bold{w}} = 0$ 我们得到$\bold{w^*}$即式$(9)$，其中$\bold{I}$为单位阵。

$$
\bold{w^*} = (\bold{X'X} + \lambda\bold{I})^{-1}\bold{X'T}\tag{9}
$$

由$(9)$实现

```python
def get_params_with_penalty(x_matrix, t_vec, lambda_penalty):
    return (
        np.linalg.pinv(
            x_matrix.T @ x_matrix + lambda_penalty * np.identity(x_matrix.shape[1])
        )
        @ x_matrix.T
        @ t_vec
    )
```

## 4、梯度下降法求解最优解

对于$f(\bold{x})$如果在$\bold{x_i}$点可微且有定义，我们知道顺着梯度 $\nabla f(\bold{x_i})$为增长最快的方向，因此梯度的反方向 $-\nabla f(\bold{x_i})$ 即为下降最快的方向。因而如果有式$(10)$对于 $\alpha > 0$ 成立,

$$
\bold{x_{i+1}}= \bold{x_i} - \alpha \nabla f(\bold{x_i}) \tag{10}
$$

那么对于序列$\bold{x_0}, \bold{x_1}, \dots$ 我们有$f(\bold{x_0}) \ge f(\bold{x_1}) \ge \dots$

> 因此，如果顺利我们可以得到一个 $f(\bold{x_n})$ 收敛到期望的最小值，对于此次实验很大可能性可以收敛到最小值。

进而我们实现算法如下，其中$\delta$为精度要求，通常可以设置为$\delta = 1\times10^{-6}$：

```python
def gradient_descent_fit(x_matrix, t_vec, lambda_penalty, w_vec_0, learning_rate=0.1, delta=1e-6):
    loss_0 = calc_loss(x_matrix, t_vec, lambda_penalty, w_vec_0)
    k = 0
    w = w_vec_0
    while True:
        w_ = w - learning_rate * calc_derivative(x_matrix, t_vec, lambda_penalty, w)
        loss = calc_loss(x_matrix, t_vec, lambda_penalty, w_)
        if np.abs(loss - loss_0) < delta:
            break
        else:
            k += 1
            if loss > loss_0:
                learning_rate *= 0.5
            loss_0 = loss
            w = w_
    return k, w
```

## 5、共轭梯度法求解最优解

> 共轭梯度法解决的主要是形如$\bold{Ax} = \bold{b}$的线性方程组解的问题，其中$\bold{A}$必须是对称的、正定的。**大概来说，共轭梯度下降就是在解空间的每一个维度分别取求解最优解的，每一维单独去做的时候不会影响到其他维**，这与梯度下降方法，每次都选泽梯度的反方向去迭代，梯度下降不能保障每次在每个维度上都是靠近最优解的，这就是共轭梯度优于梯度下降的原因。

对于第 k 步的残差$\bold{r_k = b - Ax_k}$，我们根据残差去构造下一步的搜索方向$\bold{p_k}$，初始时我们令$\bold{p_0 = r_0}$。然后利用$Gram-Schmidt$方法依次构造互相共轭的搜素方向$\bold{p_k}$，具体构造的时候需要先得到第 k+1 步的残差，即$\bold{r_{k+1} = r_k - \alpha_kAp_k}$，其中$\alpha_k$如后面的式$(11)$。

根据第 k+1 步的残差构造下一步的搜索方向$\bold{p_{k+1} = r_{k+1} + \beta_{k+1}p_k}$，其中$\beta_{k+1} = \bold{\cfrac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k}}$。

然后可以得到$\bold{x_{k+1} = x_k + \alpha_kp_k}$，其中$\alpha_k = \cfrac{\bold{p_k}^T(\bold{b} - \bold{Ax_k})}{\bold{p_k}^T\bold{Ap_k}} = \cfrac{\bold{p_k}^T\bold{r_k}}{{\bold{p_k}^T\bold{Ap_k}}}$

对于第 k 步的残差 $\bold{r_k} = \bold{b} - \bold{Ax}$，$\bold{r_k}$为$\bold{x} = \bold{x_k}$时的梯度反方向。由于我们仍然需要保证 $\bold{p_k}$ 彼此共轭。因此我们通过当前的残差和之前所有的搜索方向来构建$\bold{p_k}$，得到式$(11)$

$$
\bold{p_k} = \bold{r_k} - \sum_{i < k}\cfrac{\bold{p_i}^T\bold{Ar_k}}{\bold{p_i}^T\bold{Ap_i}}\bold{p_i}\tag{11}
$$

进而通过当前的搜索方向$\bold{p_k}$得到下一步优化解$\bold{x_{k+1}} = \bold{x_k} + \alpha_k\bold{p_k}$，其中$\alpha_k = \cfrac{\bold{p_k}^T(\bold{b} - \bold{Ax_k})}{\bold{p_k}^T\bold{Ap_k}} = \cfrac{\bold{p_k}^T\bold{r_k}}{{\bold{p_k}^T\bold{Ap_k}}}$

> 求解的方法就是我们先猜一个解$\bold{x_0}$，然后取梯度的反方向 $\bold{p_0} = \bold{b} - \bold{Ax}$，在 n 维空间的基中$\bold{p_0}$要与其与的基共轭并且为初始残差。

对于共轭梯度下降，算法实现如下，初始时取$\bold{w_0} = \left[
\begin{matrix}
    0 \\ 0 \\ \vdots \\ 0
\end{matrix}
\right]$

由上文$(8)(9)$得

$$
(\bold{X'X} + \lambda\bold{I})\bold{w^*} = \bold{X'T}\tag{12}
$$

令

$$
\left\{
\begin{array}{lr}
\bold{A} = \bold{X'X} + \lambda\bold{I} & \\
\bold{b} = \bold{X'T}
\end{array}
\right.\tag{13}
$$

```python
def switch_deri_func_for_conjugate_gradient(x_matrix, t_vec, lambda_penalty, w_vec):
    A = x_matrix.T @ x_matrix - lambda_penalty * np.identity(len(x_matrix.T))
    x = w_vec
    b = x_matrix.T @ t_vec
    return A, x, b
```

通过共轭梯度下降法求解$Ax=b$

```python
def conjugate_gradient_fit(A, x_0, b, delta=1e-6):
    x = x_0
    r_0 = b - A @ x
    p = r_0
    k = 0
    while True:
        alpha = (r_0.T @ r_0) / (p.T @ A @ p)
        x = x + alpha * p
        r = r_0 - alpha * A @ p
        if r_0.T @ r_0 < delta:
            break
        beta = (r.T @ r) / (r_0.T @ r_0)
        p = r + beta * p
        r_0 = r
        k += 1
    return k, x
```

解得$x$，并记录迭代次数$k$

# 四、实验结果分析

## 1、不带惩罚项的解析解

> 固定训练样本的大小为 15，分别使用不同多项式阶数，测试：

<center>
<figure class="half">
	<img src="https://img-blog.csdnimg.cn/20201007112200127.png" width="45%">
    <img src="https://img-blog.csdnimg.cn/20201007111458623.png" width="45%" >
</figure>
<figure class="half">
    <img src="https://img-blog.csdnimg.cn/20201007111708881.png" width="45%">
    <img src="https://img-blog.csdnimg.cn/20201007112015354.png" width="45%">
</figure>
</center>

我们可以看到在固定训练样本的大小之后，在多项式阶数为 3 时的拟合效果已经很好。继续提高多项式的阶数，尤其在阶数为 14 的时候曲线“完美的”经过了所有的节点，这种剧烈的震荡并没有很好的拟合真实的背后的函数$sin(2\pi x)$，反而将所有噪声均很好的拟合，即表现出来一种过拟合的情况。

其出现过拟合的本质原因是，在阶数过大的情况下，模型的复杂度和拟合的能力都增强，因此可以通过过大或者过小的系数来实现震荡以拟合所有的数据点，以至于甚至拟合了所有的噪声。在这里由于我们的数据样本大小只有 15，所以在阶数为 14 的时候，其对应的系数向量$\bold{w}$恰好有唯一解，因此可以穿过所有的样本点。

对于过拟合我们可以通过增加样本的数据或者通过增加惩罚项的方式来解决。增加数据集样本数量，使其超过参数向量的大小，就会在一定程度上解决过拟合问题。

> 固定多项式阶数，使用不同数量的样本数据，测试

<center>
<figure class="half">
	<img src="https://img-blog.csdnimg.cn/20201007112731914.png" width="45%">
    <img src="https://img-blog.csdnimg.cn/20201007112836739.png" width="45%" >
</figure>
<figure class="half">
    <img src="https://img-blog.csdnimg.cn/20201007113734129.png" width="45%">
    <img src="https://img-blog.csdnimg.cn/20201007113824552.png" width="45%">
</figure>
</center>

可以看到在固定多项式阶数为 7 的情况下，随着样本数量逐渐增加，过拟合的现象有所解决。特别是对比左上图与右下图的差距，可以看到样本数量对于过拟合问题是有影响的。

## 2、带惩罚项的解析解

首先根据式$(6)$我们需要确定最佳的超参数$\lambda$，因此我们通过根均方(RMS)误差来确定

<center>
<figure class="box">
	<img src="https://img-blog.csdnimg.cn/20201007122408696.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201007122525212.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201007122611185.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201007122927887.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201007123027910.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201007123156869.png" width="30%">
</figure>

观察上面的四张图， 我们可以发现对于超参数$\lambda$的选择，在$(e^{-50}, e^{-30})$左右保持一种相对稳定的错误率；但是在$(e^{-30}, e^{-5})$错误率有一个明显的下降，所以下面在下面的完整 100 次实验中我们可以看到最佳参数的分布区间也大都在这个范围内；在大于$e^{-5}$的区间内，错误率有一个急剧的升高。

> 对于不同训练集，超参数$\lambda$的选择不同。
> 因此每生成新训练集，则通过带惩罚项的解析解计算出当前的$best\_lambda$，用于带惩罚项的解析解、梯度下降和共轭梯度下降。

一般来说，最佳的超参数范围在$(e^{-10}, e^{-6})$之间。

> 比较是否带惩罚项的拟合曲线

<center>
<figure class="box">
	<img src="https://img-blog.csdnimg.cn/20201007135737279.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201007135839844.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201007135931781.png" width="30%">
</figure>

## 3、优化解

实验利用梯度下降(Gradient descent)和共轭梯度法(Conjugate gradient)两种方法来求优化解。由于该问题是有解析解存在，并且在之前的实验报告部分已经列出，所以在此处直接利用上面的解析解进行对比。**此处使用的学习率固定为$learning\_rate = 0.01$，停止的精度要求为$1 \times 10 ^ {-6}$。**

| 阶数 | 训练集规模 | 梯度下降迭代次数 | 共轭梯度下降迭代次数 |
| :--: | :--------: | :--------------: | :------------------: |
|  3   |     10     |      110939      |          4           |
|  3   |     20     |      92551       |          4           |
|  3   |     50     |      56037       |          4           |
|  5   |     10     |      38038       |          4           |
|  5   |     20     |      22867       |          5           |
|  5   |     50     |      118788      |          5           |
|  8   |     10     |      81387       |          5           |
|  8   |     20     |      71743       |          7           |
|  8   |     50     |      47445       |          8           |

首先在固定多项式阶数的情况下，**随着训练样本的增加，梯度下降的迭代次数均有所下降，但是对于共轭梯度迭代次数变化不大。**

其次在固定训练样本的情况下，**梯度下降迭代次数的变化，对于 3 阶的情况下多于 8 阶的情况对于共轭梯度的而言，迭代次数仍然较少。**

总的来说，**对于梯度下降法，迭代次数在 10000 次以上；而对于共轭梯度下降，则需要的迭代次数均不超过 10 次(即小于解空间的维度 M)。**

> 下面是不同阶数和不同训练集规模上，两种方法的比较

<center>
<figure class="box">
	<img src="https://img-blog.csdnimg.cn/20201008102151696.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201008102238708.png" width="30%" >
    <img src="https://img-blog.csdnimg.cn/20201008102502140.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201008102732771.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201008102916581.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201008103128816.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201008103231886.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/2020100810335136.png" width="30%">
    <img src="https://img-blog.csdnimg.cn/20201008103431186.png" width="30%">
</figure>
</center>

梯度下降和共轭梯度下降两种方法拟合结果相似，但共轭梯度下降收敛速度却远优于梯度下降、梯度下降稳定程度略优于共轭梯度下降。

## 4、四种拟合对比

> 3 阶，训练集规模 10

| $w$ | Analytical_Without_Penalty | Analytical_With_Penalty | Gradient_Descent | Conjugate_Gradient |
| :-: | :------------------------: | :---------------------: | :--------------: | :----------------: |
|  0  |          0.086879          |        0.086879         |     0.221466     |      0.086879      |
|  1  |         10.030641          |        10.030641        |     8.037745     |     10.030641      |
|  2  |         -29.339163         |       -29.339163        |    -24.321837    |     -29.339163     |
|  3  |         18.911480          |        18.911480        |    15.658977     |     18.911480      |

![](https://img-blog.csdnimg.cn/20201008114513238.png#pic_center)

> 5 阶，训练集规模 15

| $w$ | Analytical_Without_Penalty | Analytical_With_Penalty | Gradient_Descent | Conjugate_Gradient |
| :-: | :------------------------: | :---------------------: | :--------------: | :----------------: |
|  0  |          0.179098          |        0.179098         |     0.252970     |      0.127280      |
|  1  |          6.186679          |        6.186679         |     6.120921     |      8.802575      |
|  2  |         -4.197192          |        -4.197192        |    -14.547026    |     -24.093028     |
|  3  |         -47.562082         |       -47.562082        |    -0.720464     |      6.293567      |
|  4  |         73.581708          |        73.581708        |     6.179084     |     13.416870      |
|  5  |         -28.074476         |       -28.074476        |     2.945509     |     -4.398728      |

![](https://img-blog.csdnimg.cn/20201008113637951.png#pic_center)

> 8 阶，训练集规模 15

| $w$ | Analytical_Without_Penalty | Analytical_With_Penalty | Gradient_Descent | Conjugate_Gradient |
| :-: | :------------------------: | :---------------------: | :--------------: | :----------------: |
|  0  |          0.106875          |        -0.090665        |    -0.015363     |     -0.394036      |
|  1  |         -3.384695          |        9.291901         |     7.889612     |     18.583773      |
|  2  |         37.377871          |       -19.656246        |    -16.016235    |     -65.870523     |
|  3  |         451.762357         |        -6.583293        |    -3.968438     |     54.941302      |
|  4  |        -3775.172514        |        16.087370        |     5.885670     |     25.695923      |
|  5  |        10453.330011        |        10.996164        |     7.857581     |     -17.630151     |
|  6  |       -13828.893913        |        -3.553833        |     4.750929     |     -30.267690     |
|  7  |        8931.020178         |        -9.096546        |    -0.458928     |     -12.072342     |
|  8  |        -2266.131841        |        2.553330         |    -5.970533     |     27.074041      |

![](https://img-blog.csdnimg.cn/20201008114050851.png#pic_center)

> 15 阶，训练集规模 20

| $w$ | Analytical_Without_Penalty | Analytical_With_Penalty | Gradient_Descent | Conjugate_Gradient |
| :-: | :------------------------: | :---------------------: | :--------------: | :----------------: |
|  0  |          0.333171          |        0.056738         |     0.132317     |     -0.336080      |
|  1  |         -43.880559         |        7.449353         |     6.433572     |     14.722545      |
|  2  |        1033.334292         |       -13.398546        |    -12.067511    |     -36.803428     |
|  3  |        -8505.117304        |        -8.201093        |    -5.256489     |      3.989214      |
|  4  |        34811.021660        |        3.048536         |     2.141618     |     17.244891      |
|  5  |       -75466.068594        |        8.612557         |     5.268958     |     13.535997      |
|  6  |        75267.404153        |        8.529704         |     5.150508     |      5.089805      |
|  7  |         786.304140         |        5.220844         |     3.410453     |     -2.234687      |
|  8  |       -53324.724219        |        0.884139         |     1.230193     |     -6.603244      |
|  9  |        -2109.290031        |        -2.981579        |    -0.689982     |     -8.038983      |
| 10  |        42045.345941        |        -5.509223        |    -1.994826     |     -7.225030      |
| 11  |        14623.791181        |        -6.291021        |    -2.546300     |     -4.952217      |
| 12  |       -33191.453129        |        -5.219866        |    -2.332749     |     -1.904370      |
| 13  |       -22428.414650        |        -2.369747        |    -1.410270     |      1.399483      |
| 14  |        38870.202234        |        2.086673         |     0.133219     |      4.598064      |
| 15  |       -12368.829171        |        7.931668         |     2.199092     |      7.459104      |

![](https://img-blog.csdnimg.cn/20201008162611443.png#pic_center)

# 五、结论

- 增加训练样本的数据可以有效的解决过拟合的问题。
- 对于训练样本限制较多的问题，通过增加惩罚项仍然可以有效解决过拟合问题。
- 对于梯度下降法和共轭梯度法而言，梯度下降收敛速度较慢，共轭梯度法的收敛速度快；且二者相对于解析解而言，共轭梯度法的拟合效果解析解的效果更好。
- 相比之下，共轭梯度下降能够有效的解决梯度下降法迭代次数多，和复杂度高的有效方法。

# 六、参考文献

- [Pattern Recognition and Machine Learning.](https://www.springer.com/us/book/9780387310732)
- [Gradient descent wiki](https://en.wikipedia.org/wiki/Gradient_descent)
- [Conjugate gradient method wiki](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
- [Shewchuk J R. An introduction to the conjugate gradient method without the agonizing pain[J]. 1994](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf).

# 七、附录（代码）

> 源代码：[实验一 GitHub 仓库](https://github.com/gzn00417/2020Fall-Machine-Learning/tree/master/Labs/Lab1)

## `main.ipynb`

> Jupyter Notebook 运行效果：[主程序](https://github.com/gzn00417/2020Fall-Machine-Learning/blob/master/Labs/Lab1/main.ipynb)

```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from DataGenerator import *
from AnalyticalSolution import *
from Switcher import *
from Calculator import *
from GradientDescent import *
from ConjugateGradient import *

%matplotlib inline
sns.set(palette="Set2")
```

```python
# 超参数
TRAIN_NUM = 10 # 训练集规模
VALIDATION_NUM = 100 # 验证集规模
TEST_NUM = 1000 # 测试集规模
ORDER = 7 # 阶数
X_LEFT = 0 # 左界限
X_RIGHT = 1 # 右界限
NOISE_SCALE = 0.25 # 噪音标准差
LEARNING_RATE = 0.01 # 梯度下降学习率
DELTA = 1e-6 # 优化残差界限
W_SOLUTION = pd.DataFrame() # 不同方法的w解
LAMBDA = np.power(np.e, -7)

def base_func(x):
    """原始函数
    """
    return np.sin(2 * np.pi * x)

# 训练集
train_data = get_data(
    x_range=(X_LEFT, X_RIGHT),
    sample_num=TRAIN_NUM,
    base_func=base_func,
    noise_scale=NOISE_SCALE,
)
X_TRAIN = train_data["X"]
Y_TRAIN = train_data["Y"]

# 验证集
X_VALIDATION = np.linspace(X_LEFT, X_RIGHT, VALIDATION_NUM)
Y_VALIDATION = base_func(X_VALIDATION)

# 测试集
X_TEST = np.linspace(X_LEFT, X_RIGHT, TEST_NUM)
Y_TEST = base_func(X_TEST)

train_data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.087444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.111111</td>
      <td>0.667822</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.222222</td>
      <td>0.826025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.333333</td>
      <td>0.563550</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.444444</td>
      <td>0.548170</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.555556</td>
      <td>-0.388687</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.666667</td>
      <td>-0.803972</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.777778</td>
      <td>-1.049368</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.888889</td>
      <td>-0.648516</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.000000</td>
      <td>0.234352</td>
    </tr>
  </tbody>
</table>
</div>

### 可视化拟合结果

```python
def show_train_data(x_train, y_train):
    plt.scatter(x=x_train, y=y_train, color="white", edgecolors="darkblue")
```

```python
def show_test_data(x_test, y_test):
    plt.plot(x_test, y_test, linewidth=2, color="gray", linestyle="-.")
```

```python
def show_order_and_train_num():
    plt.title("ORDER = " + str(ORDER) + ", TRAIN_NUM = " + str(TRAIN_NUM))
```

```python
def show_comparation(x, y1, label1, y2, label2):
    """可视化两个函数
    """
    d = pd.DataFrame(
        np.concatenate(
            (
                np.transpose([X_TEST, y1, [label1] * TEST_NUM]),
                np.transpose([X_TEST, y2, [label2] * TEST_NUM]),
            )
        ),
        columns=["X", "Y", "type"],
    )
    d[["X", "Y"]] = d[["X", "Y"]].astype("float")
    sns.lineplot(x="X", y="Y", data=d, hue="type")
    show_order_and_train_num()
```

### 解析解

```python
def get_y_pred_by_analytical(x_train, y_train, x, with_penalty=False, lambda_penalty=None):
    assert not with_penalty or lambda_penalty is not None  # 有惩罚项时，lambda不为空
    x_vec, t_vec = x_train, y_train
    w_vec = (
        get_params_with_penalty(
            x_matrix=get_x_matrix(x_vec, order=ORDER),
            t_vec=t_vec,
            lambda_penalty=lambda_penalty,
        )
        if with_penalty
        else get_params(x_matrix=get_x_matrix(x_vec, order=ORDER), t_vec=t_vec)
    )
    return np.dot(get_x_matrix(x, order=ORDER), w_vec), w_vec
```

### 不带惩罚项解析解

```python
y_pred_without_penalty, w_wec_without_penalty = get_y_pred_by_analytical(
    x_train=X_TRAIN, y_train=Y_TRAIN, x=X_TEST, with_penalty=False
)
W_SOLUTION["Analytical_Without_Penalty"] = w_wec_without_penalty
```

```python
show_comparation(x=X_TEST, y1=Y_TEST, label1="True", y2=y_pred_without_penalty, label2="Pred")
show_train_data(x_train=X_TRAIN, y_train=Y_TRAIN)
```

![](https://img-blog.csdnimg.cn/20201008170919107.png#pic_center)

### 带惩罚项解析解

```python
def show_lambda_error(error_ln_lambda):
    data = pd.DataFrame(error_ln_lambda, columns=["$ln{\lambda}$", "$E_{rms}$", "type"])
    sns.lineplot(x="$ln{\lambda}$", y="$E_{rms}$", data=data, hue="type")
    idxmin = error_ln_lambda[data[data["type"]=="VALIDATION"]["$E_{rms}$"].idxmin()]
    plt.title(label=("Min: $e^{" + str(int(idxmin[0])) + "}, " + "{:.3f}".format(idxmin[1]) + "$"))
    return np.power(np.e, idxmin[0]), idxmin[1]
```

```python
error_ln_lambda = []
for i in range(-50, 0):
    y_pred, w_vec = get_y_pred_by_analytical(
        x_train=X_TRAIN,
        y_train=Y_TRAIN,
        x=X_TRAIN,  # 训练集
        with_penalty=True,
        lambda_penalty=np.exp(i),
    )
    error_ln_lambda.append(
        [i, calc_e_rms(y_pred=y_pred, y_true=Y_TRAIN), "TRAIN"]
    )  # 训练集上的根均方误差
    y_pred, w_vec = get_y_pred_by_analytical(
        x_train=X_TRAIN,
        y_train=Y_TRAIN,
        x=X_VALIDATION,  # 验证集
        with_penalty=True,
        lambda_penalty=np.exp(i),
    )
    error_ln_lambda.append(
        [i, calc_e_rms(y_pred=y_pred, y_true=Y_VALIDATION), "VALIDATION"]
    )  # 测试集上的根均方误差
```

```python
best_lambda, least_loss = show_lambda_error(error_ln_lambda)
LAMBDA = best_lambda
```

![](https://img-blog.csdnimg.cn/2020100817095059.png#pic_center)

```python
y_pred_with_penalty, w_wec_with_penalty = get_y_pred_by_analytical(
    x_train=X_TRAIN, y_train=Y_TRAIN, x=X_TEST, with_penalty=True, lambda_penalty=LAMBDA,
)
W_SOLUTION["Analytical_With_Penalty"] = w_wec_with_penalty
show_comparation(x=X_TEST, y1=Y_TEST, label1="True", y2=y_pred_with_penalty, label2="Pred")
show_train_data(x_train=X_TRAIN, y_train=Y_TRAIN)
```

![](https://img-blog.csdnimg.cn/20201008171013645.png#pic_center)

### 对比是否带惩罚项的拟合结果

```python
show_comparation(
    x=X_TEST,
    y1=y_pred_without_penalty,
    label1="Analytical_Without_Penalty",
    y2=y_pred_with_penalty,
    label2="Analytical_With_Penalty",
)
plt.legend(["Analytical_Without_Penalty", "Analytical_With_Penalty"])
```

    <matplotlib.legend.Legend at 0x24269fcaa90>

![](https://img-blog.csdnimg.cn/20201008171038691.png#pic_center)

### 梯度下降法

```python
def get_y_pred_by_gradient_descent(x_train, y_train, x, lambda_penalty):
    x_vec, t_vec = x_train, y_train
    k, w_vec = gradient_descent_fit(
        x_matrix=get_x_matrix(x_vec, order=ORDER),
        t_vec=t_vec,
        lambda_penalty=lambda_penalty,
        w_vec_0=np.zeros(ORDER + 1),
        learning_rate=LEARNING_RATE,
        delta=DELTA,
    )
    return k, np.dot(get_x_matrix(x, order=ORDER), w_vec), w_vec
```

```python
k_gradient_descent, y_pred_gradient_descent, w_wec_gradient_descent = get_y_pred_by_gradient_descent(
    x_train=X_TRAIN, y_train=Y_TRAIN, x=X_TEST, lambda_penalty=LAMBDA
)
W_SOLUTION["Gradient_Descent"] = w_wec_gradient_descent
k_gradient_descent
```

    52893

```python
show_comparation(x=X_TEST, y1=Y_TEST, label1="True", y2=y_pred_gradient_descent, label2="Pred")
show_train_data(x_train=X_TRAIN, y_train=Y_TRAIN)
```

![](https://img-blog.csdnimg.cn/20201008171105704.png#pic_center)

### 共轭梯度法

```python
def get_y_pred_by_conjugate_gradient(x_train, y_train, x, lambda_penalty):
    x_vec, t_vec = x_train, y_train
    A, x_0, b = switch_deri_func_for_conjugate_gradient(
        x_matrix=get_x_matrix(x_vec, order=ORDER),
        t_vec=t_vec,
        lambda_penalty=lambda_penalty,
        w_vec=np.zeros(ORDER + 1),
    )
    k, w_vec = conjugate_gradient_fit(A=A, x_0=x_0, b=b, delta=DELTA)
    return k, np.dot(get_x_matrix(x, order=ORDER), w_vec), w_vec
```

```python
k_conjugate_gradient, y_pred_conjugate_gradient, w_wec_conjugate_gradient = get_y_pred_by_conjugate_gradient(
    x_train=X_TRAIN, y_train=Y_TRAIN, x=X_TEST, lambda_penalty=LAMBDA,
)
W_SOLUTION["Conjugate_Gradient"] = w_wec_conjugate_gradient
k_conjugate_gradient
```

    5

```python
show_comparation(x=X_TEST, y1=Y_TEST, label1="True", y2=y_pred_conjugate_gradient, label2="Pred")
show_train_data(x_train=X_TRAIN, y_train=Y_TRAIN)
```

![](https://img-blog.csdnimg.cn/20201008171126482.png#pic_center)

### 对比梯度下降和共轭梯度的拟合结果

```python
show_comparation(
    x=X_TEST,
    y1=y_pred_gradient_descent,   label1="Gradient_Descent",
    y2=y_pred_conjugate_gradient, label2="Conjugate_Gradient",
)
show_test_data(x_test=X_TEST, y_test=Y_TEST)
plt.legend(["Gradient_Descent", "Conjugate_Gradient"])
```

    <matplotlib.legend.Legend at 0x24269d1cc88>

![](https://img-blog.csdnimg.cn/20201008171146978.png#pic_center)

### 四种拟合方法汇总

```python
W_SOLUTION
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Analytical_Without_Penalty</th>
      <th>Analytical_With_Penalty</th>
      <th>Gradient_Descent</th>
      <th>Conjugate_Gradient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.084599</td>
      <td>0.103922</td>
      <td>0.233123</td>
      <td>0.110288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.779036</td>
      <td>5.694382</td>
      <td>3.978839</td>
      <td>5.693982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-125.344309</td>
      <td>-8.782488</td>
      <td>-7.604623</td>
      <td>-8.090070</td>
    </tr>
    <tr>
      <th>3</th>
      <td>601.300442</td>
      <td>-9.708832</td>
      <td>-3.345961</td>
      <td>-12.785307</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1540.525403</td>
      <td>-1.287225</td>
      <td>0.884203</td>
      <td>2.779658</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2029.729045</td>
      <td>14.740948</td>
      <td>2.743226</td>
      <td>14.930732</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1308.321291</td>
      <td>13.958697</td>
      <td>2.525709</td>
      <td>10.353532</td>
    </tr>
    <tr>
      <th>7</th>
      <td>329.531802</td>
      <td>-14.476569</td>
      <td>0.907662</td>
      <td>-12.711493</td>
    </tr>
  </tbody>
</table>
</div>

```python
plt.figure(figsize=(10, 6))
show_train_data(x_train=X_TRAIN, y_train=Y_TRAIN)
show_test_data(x_test=X_TEST, y_test=Y_TEST)
for column in W_SOLUTION.columns:
    sns.lineplot(
        x=X_TEST,
        y=[W_SOLUTION[column] @ get_x_series(x=x, order=ORDER) for x in X_TEST],
    )
plt.legend(["$\sin (2 \pi x)$"] + list(W_SOLUTION.columns))
show_order_and_train_num()
```

![](https://img-blog.csdnimg.cn/20201008171214940.png#pic_center)

## `DataGenerator.py`

```python
import numpy as np
import pandas as pd


def get_data(
    x_range: (float, float) = (0, 1),
    sample_num: int = 10,
    base_func=lambda x: np.sin(2 * np.pi * x),
    noise_scale=0.25,
) -> "pd.DataFrame":
    X = np.linspace(x_range[0], x_range[1], num=sample_num)
    Y = base_func(X) + np.random.normal(scale=noise_scale, size=X.shape)
    data = pd.DataFrame(data=np.dstack((X, Y))[0], columns=["X", "Y"])
    return data

```

## `AnalyticalSolution.py`

```python
import numpy as np


def get_params(x_matrix, t_vec) -> [float]:
    return np.linalg.pinv(x_matrix.T @ x_matrix) @ x_matrix.T @ t_vec


def get_params_with_penalty(x_matrix, t_vec, lambda_penalty):
    return (
        np.linalg.pinv(
            x_matrix.T @ x_matrix + lambda_penalty * np.identity(x_matrix.shape[1])
        )
        @ x_matrix.T
        @ t_vec
    )

```

## `GradientDescent.py`

```python
import numpy as np
from Calculator import *


def gradient_descent_fit(
    x_matrix, t_vec, lambda_penalty, w_vec_0, learning_rate=0.1, delta=1e-6
):
    loss_0 = calc_loss(x_matrix, t_vec, lambda_penalty, w_vec_0)
    k = 0
    w = w_vec_0
    while True:
        w_ = w - learning_rate * calc_derivative(x_matrix, t_vec, lambda_penalty, w)
        loss = calc_loss(x_matrix, t_vec, lambda_penalty, w_)
        if np.abs(loss - loss_0) < delta:
            break
        else:
            k += 1
            if loss > loss_0:
                learning_rate *= 0.5
            loss_0 = loss
            w = w_
    return k, w

```

## `ConjugateGradient.py`

```python
import numpy as np


def conjugate_gradient_fit(A, x_0, b, delta=1e-6):
    """解Ax=b
    """
    x = x_0
    r_0 = b - A @ x
    p = r_0
    k = 0
    while True:
        alpha = (r_0.T @ r_0) / (p.T @ A @ p)
        x = x + alpha * p
        r = r_0 - alpha * A @ p
        if r_0.T @ r_0 < delta:
            break
        beta = (r.T @ r) / (r_0.T @ r_0)
        p = r + beta * p
        r_0 = r
        k += 1
    return k, x


```

## `Calculator.py`

```python
import numpy as np


def calc_e_rms(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def calc_loss(x_matrix, t_vec, lambda_penalty, w_vec):
    Xw_T = x_matrix @ w_vec - t_vec
    return 0.5 * np.mean(Xw_T.T @ Xw_T + lambda_penalty * w_vec @ np.transpose([w_vec]))


def calc_derivative(x_matrix, t_vec, lambda_penalty, w_vec):
    return x_matrix.T @ x_matrix @ w_vec - x_matrix.T @ t_vec + lambda_penalty * w_vec

```

## `Switcher.py`

```python
import numpy as np


def get_x_series(x, order) -> [float]:
    """return 1, x^1, x^2,..., x^n, n = order
    """
    series = [1.0]
    for _ in range(order):
        series.append(series[-1] * x)
    return series


def get_x_matrix(x_vec, order: int = 1) -> [[float]]:
    x_matrix = []
    for i in range(len(x_vec)):
        x_matrix.append(get_x_series(x_vec[i], order))
    return np.asarray(x_matrix)


def switch_deri_func_for_conjugate_gradient(x_matrix, t_vec, lambda_penalty, w_vec):
    A = x_matrix.T @ x_matrix - lambda_penalty * np.identity(len(x_matrix.T))
    x = w_vec
    b = x_matrix.T @ t_vec
    return A, x, b

```
