from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_valid_matrix = convert_matrix(valid_data)
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_valid_matrix[np.isnan(zero_valid_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    zero_valid_matrix = torch.FloatTensor(zero_valid_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return (zero_train_matrix, train_matrix, train_data, valid_data, test_data,
            zero_valid_matrix)


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
        sig = nn.Sigmoid()
        out = sig(self.h(sig(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, train_matrix, zero_train_data,
          valid_data, zero_valid_matrix, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: Dict
    :param train_matrix: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param zero_valid_matrix: 2D FloatTensor
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]
    train_losses = []
    val_losses = []
    train_accs = []
    valid_accs = []

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

        val_loss = 0.
        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(zero_valid_matrix[u]).unsqueeze(0)
            output = model(inputs)
            guess = output[0][valid_data["question_id"][i]].item()
            val_loss += ((guess - valid_data["is_correct"][i]) ** 2.)

        train_acc = evaluate(model, zero_train_data, train_data)
        valid_acc = evaluate(model, zero_train_data, valid_data)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t Validation Cost: {:.6f}\t"
              "Train Acc:{} \t Valid Acc: {}".format(epoch, train_loss,
                                                     val_loss, train_acc,
                                                     valid_acc))
    return train_losses, val_losses, train_accs, valid_accs
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    (zero_train_matrix, train_matrix, train_data, valid_data, test_data,
     zero_valid_matrix) = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 50
    lr = 0.05
    num_epoch = 8
    lamb = 0.001
    model = AutoEncoder(1774, k)
    train_losses, val_losses, train_accs, valid_accs = train(model, lr, lamb,
                                                             train_data,
                                                             train_matrix,
                                                             zero_train_matrix,
                                                             valid_data,
                                                             zero_valid_matrix,
                                                             num_epoch)

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print("Final valid acc: ", valid_accs[-1])
    print("Final test acc: ", test_acc)

    # # plot
    # x_axis = list(range(num_epoch))
    #
    # plt.figure("Cost vs epoch")
    # plt.subplot(1, 2, 1)
    # plt.title("Train Cost vs epoch")
    # plt.plot(x_axis, train_losses)
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Train Cost')
    # plt.subplot(1, 2, 2)
    # plt.title("Validation Cost vs epoch")
    # plt.plot(x_axis, val_losses)
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Validation Cost')
    # plt.show()
    #
    # plt.figure("Accuracy vs epoch")
    # plt.subplot(1, 2, 1)
    # plt.title("Train Accuracy vs epoch")
    # plt.plot(x_axis, train_accs, label='Train')
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Train Accuracy')
    # plt.subplot(1, 2, 2)
    # plt.title("Validation Accuracy vs epoch")
    # plt.plot(x_axis, valid_accs, label='Validation')
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Validation Accuracy')
    # plt.show()

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################


if __name__ == "__main__":
    main()
