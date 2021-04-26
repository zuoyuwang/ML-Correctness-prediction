from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from part_a.neural_network import train, load_data
from part_a.item_response import irt
from part_a.knn import knn_impute_by_user
import numpy as np
import torch


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        tanh = nn.Tanh()
        # sig = nn.Sigmoid()
        # out = sig(self.h(sig(self.g(inputs))))
        out = tanh(self.h(tanh(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train_nn(model, lr, lamb, train_matrix, zero_train_data,
             num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_matrix: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param num_epoch: int
    :return: Module
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            regularizer = (lamb / 2) * model.get_weight_norm()
            loss = torch.sum((output - target) ** 2.) + regularizer
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

    return model
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def weighted_error(output, train_data, weight_matrix, threshold=0.5):
    """ Evaluate the valid_data on the current model.

    :param output: 2D Matrix
    :param train_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :param weight_matrix: 2D Matrix
    :param threshold: float
    :return: float
    """

    total_w = 0
    error = 0

    for i, u in enumerate(train_data["user_id"]):
        w = weight_matrix[train_data["user_id"][i]][
            train_data["question_id"][i]]

        guess = output[train_data["user_id"][i]][
                    train_data["question_id"][i]] >= 0
        if guess != train_data["is_correct"][i]:
            error += w
        total_w += w
    return error / total_w


def boost(iterations, train_data, train_matrix, val_data):
    """
    :param iterations: int
    :param train_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :param train_matrix: 2D matrix
    :param val_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: tuple
    """
    # initialize weight matrix
    N, M = train_matrix.shape
    weight = train_matrix.copy()
    weight[~np.isnan(weight)] = 1

    # hyperparameter for neural network
    k = 10
    lr = 0.05
    num_epoch = 1
    lamb = 0.05

    v_acc_list = []
    t_acc_list = []
    # inputs for prediction
    input_data = train_matrix.copy()
    input_data[np.isnan(input_data)] = 0
    input_data = torch.FloatTensor(input_data)
    inputs = Variable(input_data).unsqueeze(0)
    final = np.zeros([N, M])
    for i in range(iterations):
        # weighted data and covert to neural network model input
        weight_data = np.multiply(weight, train_matrix)
        zero_weight_data = weight_data.copy()
        zero_weight_data[np.isnan(zero_weight_data)] = 0
        zero_weight_data = torch.FloatTensor(zero_weight_data)
        weight_data = torch.FloatTensor(weight_data)
        model = AutoEncoder(1774, k)

        # train model and generate output convert to 2D nparray

        model = train_nn(model, lr, lamb, weight_data, zero_weight_data,
                         num_epoch)

        output = model(inputs)[0].cpu().data.numpy()

        error = weighted_error(output, train_data, weight)
        coe = 0.5 * np.log((1 - error) / error)
        weight = np.multiply(weight, np.exp(-coe * train_matrix * output))
        final += coe * output

        tacc = sparse_matrix_evaluate(train_data, final, threshold=0)
        vacc = sparse_matrix_evaluate(val_data, final, threshold=0)
        v_acc_list.append(vacc)
        t_acc_list.append(tacc)
        print(
            "Iteration: {}\t train_acc:{} \t, val_cc: {}".format(i, tacc, vacc))

    return final, v_acc_list, t_acc_list


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # convert 0 entries to -1 to perform boost algorithm
    sparse_matrix[np.where(sparse_matrix == 0)] = -1

    iteration = 20
    output, v_acc_list, t_acc_list = boost(iteration, train_data, sparse_matrix,
                                  val_data)
    # _, t_acc_list, _ = boost(iteration, train_data, sparse_matrix, test_data)

    val_acc = sparse_matrix_evaluate(val_data, output, threshold=0)
    test_acc = sparse_matrix_evaluate(test_data, output, threshold=0)
    print("Val_acc:", val_acc)
    print("Test_acc:", test_acc)

    plt.title("Accuracy vs iteration")
    x_axis = range(iteration)
    plt.plot(x_axis, t_acc_list, label="train")
    plt.plot(x_axis, v_acc_list, label="validation")
    plt.legend()
    plt.xlabel('Number of iteration')
    plt.ylabel('Accuracy')
    plt.show()

    # zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    # nn_model = AutoEncoder(1774, k = 10)
    # _, nn_val_acc = train(nn_model, 0.05, 0.5, train_matrix, zero_train_matrix, valid_data, iteration)
    # _, nn_test_acc = train(nn_model, 0.05, 0.5, train_matrix, zero_train_matrix, test_data, iteration)
    #
    # _, _, irt_val_acc, _, _, _ = irt(train_data, val_data, 0.05, 10)
    # _, _, irt_test_acc, _, _, _ = irt(train_data, test_data, 0.05, 10)

    # knn_val_acc = knn_impute_by_user(train_matrix, valid_data, 11)
    # knn_test_acc = knn_impute_by_user(train_matrix, test_data, 11)

    # plot validation accuracy as function of k
    # plt.title("Validation Accuracies")
    # plt.plot(v_acc_list)
    # plt.plot(nn_val_acc)
    # plt.plot(irt_val_acc)
    # plt.plot(np.repeat(knn_val_acc, iteration))
    # plt.xlabel('Number of iteration')
    # plt.ylabel('Validation Accuracy')
    # plt.legend(["Our model", "Neural Network", "Item Response Theory", "KNN"])
    # plt.show()
    # plt.title("Test Accuracies")
    # plt.plot(t_acc_list)
    # plt.plot(nn_test_acc)
    # plt.plot(irt_test_acc)
    # plt.plot(np.repeat(knn_test_acc, iteration))
    # plt.xlabel('Number of iteration')
    # plt.ylabel('Test Accuracy')
    # plt.legend(["Our model", "Neural Network", "Item Response Theory", "KNN"])
    # plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
