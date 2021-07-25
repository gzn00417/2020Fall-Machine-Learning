# 朴素贝叶斯和逻辑回归

## 贝叶斯法则
![](https://img-blog.csdnimg.cn/20201117100213655.png#pic_center)
## 条件独立
- 如果P(X,Y|Z)=P(X|Z)P(Y|Z)，或等价地P(X|Y,Z）=P(X|Z），则称事件X,Y对于给定事件Z是条件独立的，也就是说，**当Z发生时，X发生与否与Y发生与否是无关的**。

![](https://img-blog.csdnimg.cn/20201117100213709.png#pic_center)

## 朴素贝叶斯

> 假设每个输入变量独立

![](https://img-blog.csdnimg.cn/20201117100213727.png#pic_center)

### 算法
![](https://img-blog.csdnimg.cn/20201117100213719.png#pic_center)


- 概率必须满足归一性，所以只需要估计n-1个参数

## 高斯朴素贝叶斯 GNB
- 连续$X$
- 离散$Y$

### 算法
![](https://img-blog.csdnimg.cn/20201117165909108.png#pic_center)
### 参数估计
![](https://img-blog.csdnimg.cn/20201117171054374.png#pic_center)