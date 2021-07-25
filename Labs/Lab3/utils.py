import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from itertools import permutations


def generate_data(k, n, dimension, mu_list, sigma_list):
    """
    使用高斯分布产生一组数据，共k个高斯分布，每个分布产生n个数据
    得到的是带有真实类别标签的X，以便于绘图
    """
    X = np.zeros((n * k, dimension + 1))
    for i in range(k):
        X[i * n : (i + 1) * n, :dimension] = np.random.multivariate_normal(
            mu_list[i], sigma_list[i], size=n
        )
        X[i * n : (i + 1) * n, dimension : dimension + 1] = i
    return X


def kmeans(X, k, epsilon=1e-5):
    """
    K-means算法实现，算得分类结果和中心
    """
    center = np.zeros((k, X.shape[1] - 1))
    for i in range(k):
        center[i, :] = X[np.random.randint(0, high=X.shape[0]), :-1]
    iterations = 0
    while True:
        iterations += 1
        distance = np.zeros(k)
        # 根据中心重新给每个点贴分类标签
        for i in range(X.shape[0]):
            for j in range(k):
                distance[j] = np.linalg.norm(X[i, :-1] - center[j, :])
            X[i, -1] = np.argmin(distance)
        # 根据每个点新的标签计算它的中心
        new_center = np.zeros((k, X.shape[1] - 1))
        count = np.zeros(k)
        for i in range(X.shape[0]):
            new_center[int(X[i, -1]), :] += X[i, :-1]  # 对每个类的所有点坐标求和
            count[int(X[i, -1])] += 1
        for i in range(k):
            new_center[i, :] = new_center[i, :] / count[i]  # 对每个类的所有点坐标求平均值
        if np.linalg.norm(new_center - center) < epsilon:  # 用差值的二范数表示精度
            break
        else:
            center = new_center
    return X, center, iterations


def e_step(x, mu_list, sigma_list, pi_list):
    """
    e步，求每个样本由各个混合高斯成分生成的后验概率
    """
    k = mu_list.shape[0]
    gamma_z = np.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        pi_times_pdf_sum = 0
        pi_times_pdf = np.zeros(k)
        for j in range(k):
            pi_times_pdf[j] = pi_list[j] * multivariate_normal.pdf(
                x[i], mean=mu_list[j], cov=sigma_list[j]
            )
            pi_times_pdf_sum += pi_times_pdf[j]
        for j in range(k):
            gamma_z[i, j] = pi_times_pdf[j] / pi_times_pdf_sum
    return gamma_z


def m_step(x, mu_list, gamma_z):
    """
    m步，根据公式更新参数
    """
    k = mu_list.shape[0]
    n = x.shape[0]
    dim = x.shape[1]
    mu_list_new = np.zeros(mu_list.shape)
    sigma_list_new = np.zeros((k, dim, dim))
    pi_list_new = np.zeros(k)
    for j in range(k):
        n_j = np.sum(gamma_z[:, j])
        pi_list_new[j] = n_j / n  # 计算新的pi

        gamma = gamma_z[:, j]
        gamma = gamma.reshape(n, 1)
        mu_list_new[j, :] = (gamma.T @ x) / n_j  # 计算新的mu
        sigma_list_new[j] = (
            (x - mu_list[j]).T @ np.multiply((x - mu_list[j]), gamma)
        ) / n_j  # 计算新的sigma
    return mu_list_new, sigma_list_new, pi_list_new


def log_likelihood(x, mu_list, sigma_list, pi_list):
    """
    计算极大似然对数
    """
    log_l = 0
    for i in range(x.shape[0]):
        pi_times_pdf_sum = 0
        for j in range(mu_list.shape[0]):
            pi_times_pdf_sum += pi_list[j] * multivariate_normal.pdf(
                x[j], mean=mu_list[j], cov=sigma_list[j]
            )
        log_l += np.log(pi_times_pdf_sum)
    return log_l


def gmm(X, k, epsilon=1e-5):
    """
    GMM算法 
    """
    x = X[:, :-1]
    pi_list = np.ones(k) * (1.0 / k)
    sigma_list = np.array([0.1 * np.eye(x.shape[1])] * k)
    # 随机选第1个初始点，依次选择与当前mu中样本点距离最大的点作为初始簇中心点
    mu_list = [x[np.random.randint(0, k) + 1]]
    for times in range(k - 1):
        temp_ans = []
        for i in range(x.shape[0]):
            temp_ans.append(
                np.sum([np.linalg.norm(x[i] - mu_list[j]) for j in range(len(mu_list))])
            )
        mu_list.append(x[np.argmax(temp_ans)])
    mu_list = np.array(mu_list)

    old_log_l = log_likelihood(x, mu_list, sigma_list, pi_list)
    iterations = 0
    log_l_list = pd.DataFrame(columns=("Iterations", "log likelihood"))
    while True:
        gamma_z = e_step(x, mu_list, sigma_list, pi_list)
        mu_list, sigma_list, pi_list = m_step(x, mu_list, gamma_z)
        new_log_l = log_likelihood(x, mu_list, sigma_list, pi_list)
        if iterations % 10 == 0:
            log_l_list = log_l_list.append(
                [{"Iterations": iterations, "log likelihood": old_log_l}],
                ignore_index=True,
            )
        if old_log_l < new_log_l and (new_log_l - old_log_l) < epsilon:
            break
        old_log_l = new_log_l
        iterations += 1
    # 计算标签
    for i in range(X.shape[0]):
        X[i, -1] = np.argmax(gamma_z[i, :])
    return X, iterations, log_l_list


def get_accuracy(real_lable, class_lable, k):
    """
    计算聚类准确率
    """
    classes = list(permutations(range(k), k))
    counts = np.zeros(len(classes))
    for i in range(len(classes)):
        for j in range(real_lable.shape[0]):
            if int(real_lable[j]) == classes[i][int(class_lable[j])]:
                counts[i] += 1
    return np.max(counts) / real_lable.shape[0]


def get_result_kmeans(X, k, real_lable, epsilon):
    X, center, iterations = kmeans(X, k, epsilon=epsilon)
    print(center)
    accuracy = get_accuracy(real_lable, X[:, -1], k)
    show(
        X,
        center,
        title="epsilon="
        + str(epsilon)
        + ", iterations="
        + str(iterations)
        + ", accuracy="
        + str(accuracy),
    )


def get_result_gmm(X, k, real_lable, epsilon):
    X, iterations, log_l_list = gmm(X, k, epsilon=epsilon)
    print(log_l_list)
    accuracy = get_accuracy(real_lable, X[:, -1], k)
    show(
        X,
        title="epsilon="
        + str(epsilon)
        + ", iterations="
        + str(iterations)
        + ", accuracy="
        + str(accuracy),
    )


def show(X, center=None, title=None):
    plt.style.use("seaborn")
    plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], marker=".", s=40, cmap="YlGnBu")
    if not center is None:
        plt.scatter(center[:, 0], center[:, 1], c="r", marker="x", s=250)
    if not title is None:
        plt.title(title)
    plt.show()


def uci_iris():
    """
    读取iris数据集
    """
    data_set = pd.read_csv("./iris.csv")
    classes = data_set["class"]
    X = np.zeros(data_set.shape)
    X[:, :-1] = np.array(data_set.drop("class", axis=1), dtype=float)
    for i in range(classes.shape[0]):
        if classes[i] == "Iris-setosa":
            continue
        elif classes[i] == "Iris-versicolor":
            X[i, -1] = 1
        elif classes[i] == "Iris-virginica":
            X[i, -1] = 2
    return X
