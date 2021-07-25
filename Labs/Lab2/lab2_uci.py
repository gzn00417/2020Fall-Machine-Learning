import numpy as np
import io
import re
import operator


def sigmoid(inx):
    return 1 / (1 + np.exp(-inx))


def model(X, W):
    return sigmoid(np.dot(X, W))


def cal_loss(W, X, Y):
    cost = -np.dot(Y.T, np.log(model(X, W))) - np.dot(
        (np.ones((scale_of_example * 2, 1)) - Y).T,
        np.log(np.ones((scale_of_example * 2, 1)) - model(X, W)),
    )
    cost = sum(cost) / len(X) + (hyper_parameter * (np.dot(W.T, W))) / (2 * len(X))
    print(cost)
    return cost


def cal_gradient(W, X, Y, lamb):
    gradient = np.dot(X.T, (model(X, W) - Y))
    return gradient + lamb * W


def gradient_decent(W, example_added, label, threshold):
    alpha = 0.0001
    gradient = cal_gradient(W, example_added, label, 0.05)
    while cal_loss(W, example_added, label) > threshold:
        W = W - alpha * gradient
        gradient = cal_gradient(W, example_added, label, 0.05)
    return W


def judge(W):
    judge_scale = 50
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
    bl = io.open("balance-scale.data", encoding="UTF-8")
    bl_list = bl.readlines()
    scale_of_example = 200
    dim = 4
    example = [1, 1, 1, 1, 1]
    label = []
    i = 0
    """
    二分类只选取左倾或者右倾的数据
    平衡状态不考虑
    选取前400个数据训练
    后一百个数据测试
    """
    for line in bl_list:
        l = re.split("[,\n]", line)
        if operator.eq(l[0], "L"):
            label.append(0)
            tmp = np.mat([1, int(l[1]), int(l[2]), int(l[3]), int(l[4])])
            example = np.vstack((example, tmp))
        elif operator.eq(l[0], "R"):
            label.append(1)
            tmp = np.mat([1, int(l[1]), int(l[2]), int(l[3]), int(l[4])])
            example = np.vstack((example, tmp))
        else:
            continue
        i = i + 1

    example_added = example[1:401, :]
    test_set = example[401:501, :]
    training_label = np.mat(label).T[1:401, :]
    test_label = np.mat(label).T[401:501, :]
    label = training_label
    W = np.zeros((dim + 1, 1))
    hyper_parameter = 0.00001
    cal_loss(W, example_added, label)
    W = gradient_decent(W, example_added, label, 0.402)
    print(judge(W))
