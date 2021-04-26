# TODO: complete this file.
from sklearn.impute import KNNImputer
from utils import *
from part_a.neural_network import AutoEncoder
from part_a.item_response import update_theta_beta, sigmoid
import torch
from sklearn.utils import resample
from torch.autograd import Variable


def knn(matrix, k):
    # resample
    matrix = resample(matrix)
    nbrs = KNNImputer(n_neighbors=k)
    # train the model
    mat = nbrs.fit_transform(matrix)
    return mat


def neural_network(train_matrix, lr, num_epoch):
    # resample
    train_matrix = resample(train_matrix)
    # build parameters
    model = AutoEncoder(1774, 50)
    zero_train_data = train_matrix.copy()
    zero_train_data[np.isnan(train_matrix)] = 0
    zero_train_data = torch.FloatTensor(zero_train_data)
    train_matrix = torch.FloatTensor(train_matrix)
    # train the model
    nn_train(model, lr, train_matrix, num_epoch, zero_train_data)

    inputs = Variable(zero_train_data).unsqueeze(0)
    output = model(inputs)[0].cpu().data.numpy()
    return output


# from neural_network.py
def nn_train(model, lr, train_matrix, num_epoch, zero_train_data):
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]
    for epoch in range(0, num_epoch):

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            loss = torch.sum((output - target) ** 2.)
            loss.backward()
            optimizer.step()


def irt(train_matrix, beta, theta, lr, iterations):
    N, M = train_matrix.shape
    pred = np.zeros((N, M))
    # resample
    train_matrix = resample(train_matrix)
    # train theta and beta
    for num in range(iterations):
        theta, beta = update_theta_beta(train_matrix, lr, theta, beta)
    # predictions
    for i in range(N):
        for j in range(M):
            pred[i][j] = sigmoid(theta[i] - beta[j])

    return pred


def ensemble(matrix, test_data):
    N, M = matrix.shape
    theta = np.random.rand(N, 1)
    beta = np.random.rand(M, 1)

    # knn
    knn_pred = knn(matrix, 11)
    # neural network
    nn_pred = neural_network(matrix, 0.05, 8)
    # item response theory
    irt_pred = irt(matrix, beta, theta, 0.01, 12)

    final_pred = (knn_pred + nn_pred + irt_pred) / 3
    final_acc = sparse_matrix_evaluate(test_data, final_pred)
    return final_acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    # train_data = load_train_csv("../data")
    test_data = load_public_test_csv("../data")
    val_data = load_valid_csv("../data")
    final_val_acc = ensemble(sparse_matrix, val_data)
    final_test_acc = ensemble(sparse_matrix, test_data)
    print(f"final validation accuracy = {final_val_acc}")
    print(f"final test accuracy = {final_test_acc}")


if __name__ == "__main__":
    main()
