import matplotlib.pyplot as plt
import numpy as np


def sigmoid(inx):
    return 1 / (1 + np.exp(-inx))


def model(X, W):
    """
    预测函数
    """
    return sigmoid(np.dot(X, W))


def cal_loss(W, X, Y):
    """
    cost1为对数似然函数
    cost2为损失函数
    """
    cost = -np.dot(Y.T, np.log(model(X, W))) - np.dot(
        (np.ones((scale_of_example * 2, 1)) - Y).T,
        np.log(np.ones((scale_of_example * 2, 1)) - model(X, W)),
    )
    cost = sum(cost) / len(X) + (hyper_parameter * (np.dot(W.T, W))) / (2 * len(X))
    print(cost)
    return cost


def cal_gradient(W, X, Y, lamb):
    """
    计算梯度
    """
    gradient = np.dot(X.T, (model(X, W) - Y))
    # print(gradient)
    return gradient + lamb * W


def gradient_decent(W, example_added, label, threshold):
    """
    梯度下降法
    """
    alpha = 0.001
    gradient = cal_gradient(W, example_added, label)
    while cal_loss(W, example_added, label) > threshold:
        W = W - alpha * gradient
        gradient = cal_gradient(W, example_added, label)
    return W


def Hessian(W, X, Y):
    """
    生成黑塞矩阵
    """
    hessianMatrix = np.zeros((dim + 1, dim + 1))
    for t in range(scale_of_example * 2):
        X_mat = np.mat(X[t]).T
        XXT = np.array(X_mat * X_mat.T)
        hessianMatrix += sigmoid(np.dot(X[t], W)) * (sigmoid(np.dot(X[t], W)) - 1) * XXT
    return hessianMatrix


def newton(W, example_added, label, threshold, step=10):
    """
    牛顿法
    """
    gradient = cal_gradient(W, example_added, label, 0.01)
    alpha = 0.01
    while cal_loss(W, example_added, label) > threshold:
        H = np.linalg.inv(Hessian(W, example_added, label))
        W = W + alpha * np.dot(H, gradient)
        gradient = cal_gradient(W, example_added, label, 0.01)
    return W


def judge(W):
    """
    判断回归效果 
    """
    judge_scale = 500
    s1 = np.dot(np.random.randn(judge_scale, dim), R1) + mu1
    plt.plot(s1[:, 0], s1[:, 1], "+", label="test_set1", color="b")
    s2 = np.dot(np.random.randn(judge_scale, dim), R2) + mu2
    plt.plot(s2[:, 0], s2[:, 1], "+", label="test_set2", color="g")
    example = np.vstack((s1, s2))
    label1 = np.zeros((judge_scale, 1))
    label2 = np.ones((judge_scale, 1))
    test_label = np.vstack((label1, label2))
    test_set = np.hstack((np.ones((judge_scale * 2, 1)), example))
    result = np.zeros((judge_scale * 2, 1))
    correct_num = 0
    for i in range(judge_scale * 2):
        if model(test_set, W)[i - 1][0] > 0.5:
            result[i - 1][0] = 1
    for i in range(judge_scale * 2):
        if result[i - 1][0] == test_label[i - 1][0]:
            correct_num += 1
    return correct_num / (judge_scale * 2)


if __name__ == "__main__":
    """
    生成训练数据，为二维随机高斯分布
    label为二分类分别为0和1
    hyper_parameter为\lambda
    """
    scale_of_example = 1000
    dim = 2
    mu1 = np.array([[1, 3]])
    Sigma1 = np.array([[2, -1], [-1, 2]])
    R1 = np.linalg.cholesky(Sigma1)
    s1 = np.dot(np.random.randn(scale_of_example, dim), R1) + mu1
    # plt.plot(s1[:, 0], s1[:, 1], ".", label="training_set1", color="red")

    mu2 = np.array([[3, 6]])
    Sigma2 = np.array([[2, -1], [-1, 2]])
    R2 = np.linalg.cholesky(Sigma2)
    s2 = np.dot(np.random.randn(scale_of_example, dim), R2) + mu2
    # plt.plot(s2[:, 0], s2[:, 1], ".", label="training_set2", color="yellow")

    example = np.vstack((s1, s2))
    label1 = np.zeros((scale_of_example, 1))
    label2 = np.ones((scale_of_example, 1))
    label = np.vstack((label1, label2))
    data = np.hstack((example, label))
    W = np.ones((dim + 1, 1))

    hyper_parameter = 0.01

    example_added = np.hstack((np.ones((scale_of_example * 2, 1)), example))
    cal_loss(W, example_added, label)

    # W = gradient_decent(W, example_added, label, 0.1)     # 梯度下降法
    W = newton(W, example_added, label, 0.1)  # 牛顿法
    print(judge(W))
    X1 = np.linspace(-2, 8, 20)
    X2 = -W[0][0] / W[2][0] - np.dot(W[1][0], X1) / W[2][0]
    plt.plot(X1, X2, label="newton_method_with_regular_term", color="m")
    plt.title(
        "scale of training examples=1000, scale of test examples=500, threshold = 0.1"
    )
    plt.legend()
    plt.show()
