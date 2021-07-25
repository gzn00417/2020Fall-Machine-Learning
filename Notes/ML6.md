# 无监督学习
> “Learning from unlabeled/unannotated data” (without supervision)

# 聚类概念
> the process of grouping a set of objects into classes of similar objects

1. 定义“类”
2. 定义“相似”、“距离”
3. 表示：向量
4. 簇数
5. 聚类算法
6. 形式基础与收敛性

# 相似度函数
计算两个数据点的“相似性”

**欧式距离**——向量空间

# 层次聚类
层次聚类，是一种很直观的算法。顾名思义就是要一层一层地进行聚类，可以从下而上地把小的cluster合并聚集，也可以从上而下地将大的cluster进行分割。似乎一般用得比较多的是从下而上地聚集，因此这里我就只介绍这一种。

所谓从下而上地合并cluster，具体而言，就是每次找到距离最短的两个cluster，然后进行合并成一个大的cluster，直到全部合并为一个cluster。整个过程就是建立一个树结构，类似于下图。

![](https://img-blog.csdnimg.cn/20201119104726253.png#pic_center)

最近的两类
![](https://img-blog.csdnimg.cn/20201119151330785.png#pic_center)



# K-means聚类
1. 选取K个点做为初始聚集的簇心（也可选择非样本点）;
2. 分别计算每个样本点到 K个簇核心的距离（这里的距离一般取欧氏距离或余弦距离），找到离该点最近的簇核心，将它归属到对应的簇；
3. 所有点都归属到簇之后， M个点就分为了 K个簇。之后重新计算每个簇的重心（平均距离中心），将其定为新的“簇核心”；
4. 反复迭代 2 - 3 步骤，直到达到某个中止条件。

![](https://img-blog.csdnimg.cn/20201119152657406.png#pic_center)