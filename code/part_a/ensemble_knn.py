# TODO: complete this file.
from utils import *
from sklearn.impute import KNNImputer
from sklearn.utils import resample


def knn_impute_by_user(matrix, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D sparse matrix
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    return mat


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    val_acc = []
    model = []
    k = 11
    # train 3 knn models with resampled matrix
    for i in range(3):
        matrix = resample(sparse_matrix)
        mat = knn_impute_by_user(matrix, k)
        acc = sparse_matrix_evaluate(val_data, mat)
        val_acc.append(acc)
        print("Validation Accuracy: {}".format(acc))
        model.append(mat)

    final_model = sum(model) / len(model)
    final_val = sparse_matrix_evaluate(val_data, final_model)
    final_test = sparse_matrix_evaluate(test_data, final_model)
    print("Final validation accuracy: ", final_val)
    print("Final test accuracy: ", final_test)


if __name__ == "__main__":
    main()
