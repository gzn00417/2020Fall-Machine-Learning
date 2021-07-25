# 机器学习中的概率论
![](https://img-blog.csdnimg.cn/20201115104646999.png#pic_center)

## 随机变量
略
## 离散型概率分布
略
## 连续性概率分布
- Normal (Gaussian) Probability Density Function
	- pdf$$p(x)=\frac{1}{\sqrt{2\pi}\delta}e^{-\frac{(x-\mu)^2}{2\delta^2}}$$
	- cdf
- Exponential Probability Distribution
	- pdf$$p(x)=\frac{1}{\mu}e^{-\frac{x}{\mu}}$$
	- cdf$$P(x\le x_0)=1-e^{-\frac{x_0}{\mu}}$$

## 独立
$$P(A ∩ B) = P(A) * P(B)$$


## 条件概率
$$P(A ∩ B) = P(A|B) P(B)$$

## 条件独立
$$P(A ∩ B) = P(A) * P(B) ≡ P(A|B) = P(A)$$
$$P(A ∩ B|C) = P(A|C) * P(B|C) ≡ P(A|B,C) = P(A|C)$$
## 先验概率 & 后验概率
Suppose that our propositions have a "causal flow"
![](https://img-blog.csdnimg.cn/20201113202424475.png#pic_center)
- 命题的先验或无条件概率
e.g., P(Flu) = 0.025 and P(DrinkBeer ) = 0.2
在任何（新的）证据到达之前与信念一致
- 命题的后验概率或条件概率
e.g., P(Headache|Flu) = 0.5 and P(Headache|Flu,DrinkBeer ) = 0.7
与新证据到达后更新的信念相对应
## 概率推理
> H = "having a headache"
> F = "coming down with Flu"
> - P(H)=1/10
> - P(F)=1/40
> - P(H|F)=1/2

- Question: $P(F|H)$
- Answer: $$P(F|H) = \frac{P(F\cap H)}{P(F)} = \frac{P(H) * P(H|F)}{P(F)} = \frac{1}{8}$$

## 贝叶斯法则
$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$
$$P(Y=y|X)=\frac{P(X|Y=y)P(Y=y)}{\Sigma_yP(X|Y=y)P(Y=y)}$$

## 联合概率 & 边缘概率
> Joint and Marginal Probabilities

略

## 最大似然估计 MLE & 最大后验概率估计 MAP
最大似然估计 MLE：给定一堆数据，假如我们知道它是从某一种分布中随机取出来的，可是我们并不知道这个分布具体的参，即“**模型已定，参数未知**”。 MLE就可以用来估计模型的参数。MLE的目标是找出一组参数，使得模型产生出观测数据的概率最大
![](https://img-blog.csdnimg.cn/20201115102741627.png#pic_center)
最大后验概率估计MAP是贝叶斯学派常用的估计方法。假设数据是i.i.d.的一组抽样。那么MAP对的估计方法可以如下推导：
![](https://img-blog.csdnimg.cn/20201115103925803.png#pic_center)

