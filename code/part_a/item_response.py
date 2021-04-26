from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(matrix, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param matrix: 2D matrix
    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    # convert to nparray and ignore nan values

    N, M = matrix.shape
    first_term = np.nansum(matrix, axis=1, keepdims=1).T.dot(theta)
    second_term = np.nansum(matrix, axis=0, keepdims=1).dot(beta)
    third_term = 0
    for i in range(N):
        for j in range(M):
            if not np.isnan(matrix[i][j]):
                third_term += np.log(1 + np.exp(theta[i] - beta[j]))
    log_lklihood = float(first_term - second_term - third_term)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param matrix: 2D matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    N, M = matrix.shape
    # update theta
    d_theta_first = np.nansum(matrix, axis=1, keepdims=1)
    d_theta_second = np.zeros([N, 1])
    for i in range(N):
        # find all non-nan entries
        all_beta = beta[~np.isnan(matrix[i])]
        d_theta_second[i] = np.sum(sigmoid(theta[i] - all_beta))
    d_theta = - d_theta_first + d_theta_second
    theta = theta - lr * d_theta

    # update beta
    d_beta_first = np.nansum(matrix, axis=0, keepdims=1).T
    d_beta_second = np.zeros([M, 1])
    for j in range(M):
        # find all non-nan entries
        all_theta = theta[~np.isnan(matrix.T[j])]
        d_beta_second[j] = np.sum(sigmoid(all_theta - beta[j]))
    d_beta = d_beta_first - d_beta_second
    beta = beta - lr * d_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_matrix, val_matrix, train_data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param train_matrix: 2D matrix
    :param val_matrix: 2D matrix
    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # Initialize theta and beta.
    N, M = train_matrix.shape
    theta = np.random.uniform(0, 1, (N, 1))
    beta = np.random.uniform(0, 1, (M, 1))

    val_acc_lst = []
    train_acc_lst = []
    train_log_lst = []
    val_log_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(train_matrix, theta=theta, beta=beta)
        neg_lld_v = neg_log_likelihood(val_matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        score2 = evaluate(data=train_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_acc_lst.append(score2)
        train_log_lst.append(neg_lld)
        val_log_lst.append(neg_lld_v)
        print("NLLK: {} \t Val_Score: {} \t Train_Score: {}".format(neg_lld,
                                                                    score,
                                                                    score2))
        theta, beta = update_theta_beta(train_matrix, lr, theta, beta)

    return theta, beta, val_acc_lst, train_acc_lst, train_log_lst, val_log_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    # convert to 2D np-array and ignore nan values
    matrix = sparse_matrix.toarray()

    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    val_matrix = convert_matrix(val_data)
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 12
    theta, beta, val_acc_lst, train_acc_lst, train_log_lst, val_log_lst = irt(
        matrix, val_matrix, train_data, val_data, lr, iterations)

    # plots (b)
    plt.figure("Accuracy vs iteration")
    x_axis = list(range(iterations))
    plt.plot(x_axis, train_acc_lst, label='Train accuracy')
    plt.plot(x_axis, val_acc_lst, label='Validation accuracy')
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure("Negative log likelihood vs iteration")
    plt.plot(x_axis, train_log_lst, label='Train log likelihood')
    plt.plot(x_axis, val_log_lst, label='Validation log likelihood')
    plt.xlabel('Number of iterations')
    plt.ylabel('Negative log likelihood')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # part (c)
    print('Final validation accuracy: ', evaluate(val_data, theta, beta))
    print('Final test accuracy: ', evaluate(test_data, theta, beta))
    #####################################################################

    # part (d)
    plt.figure("P(Cij) for five questions")
    for j in range(5):
        # find all non-nan entries
        b = beta[j]
        all_theta = theta[~np.isnan(matrix.T[j])]
        predict = sigmoid(all_theta - b)
        plt.plot(all_theta, predict, label="Question {}".format(j))
    plt.xlabel('Theta')
    plt.ylabel('Predicted probability')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
