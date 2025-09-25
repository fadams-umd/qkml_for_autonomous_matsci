import numpy as np
from ast import literal_eval
import json
from scipy.io import loadmat
from scipy.spatial.distance import cosine as cosine_distance
import gpflow

DATA_INDICES = np.array([
    25, 68, 112, 161,
    37, 91, 136, 184,
    77, 85, 128, 224,
    168, 229, 220, 233,
    195, 210, 256, 264,
])


def get_quantum_kernel_matrix_from_file(filename):
    with open(filename) as f:
        data = json.loads(f.read())

    num_points = 20

    kernel_matrix = np.ones((num_points, num_points), dtype=float)

    for k, v in data["kernel_entries"].items():
        i, j = literal_eval(k)

        try:
            kernel_matrix[i, j] = v["result"]["0"]
            kernel_matrix[j, i] = v["result"]["0"]
        except KeyError:
            kernel_matrix[i, j] = 0
            kernel_matrix[j, i] = 0

    return kernel_matrix


def get_measured_quantum_kernel_matrix():
    return get_quantum_kernel_matrix_from_file(
        "kernel_qpu_test_results/kernel_qpu_test_results_240229.json"
    )


def get_simulated_quantum_kernel_matrix():
    return get_quantum_kernel_matrix_from_file(
        "kernel_qpu_test_results/kernel_cpu_test_results_240829.json"
    )


def get_xrd_data():
    return loadmat("FeGaPd_full_data_220104a.mat")["X"][
        np.ix_(DATA_INDICES, np.arange(631, 1181))
    ]


def get_kernel_matrix_from_kernel_function(kernel_function, *args):
    xrd = get_xrd_data()

    num_points = xrd.shape[0]

    kernel_matrix = np.ones([num_points] * 2)

    for i in range(num_points):
        for j in range(i + 1, num_points):
            value = kernel_function(xrd[i, :], xrd[j, :], *args)

            kernel_matrix[i, j] = value
            kernel_matrix[j, i] = value

    return kernel_matrix


def angular_rbf_kernel_function(x, y):
    return np.exp(-(np.arccos(1 - cosine_distance(x, y)) ** 2))


def cosine_kernel_function(x, y):
    return 1 - cosine_distance(x, y)


def cosine_rbf_kernel_function(x, y):
    return np.exp(-(cosine_distance(x, y) ** 2))


def cosine_distance_exponential_kernel_function(x, y):
    return np.exp(-cosine_distance(x, y))


def gaussian_rbf_euclidean_kernel_function(x, y, lengthscale):
    return np.exp(-((np.linalg.norm(x - y) / lengthscale) ** 2))


def get_angular_rbf_kernel_matrix():
    return get_kernel_matrix_from_kernel_function(angular_rbf_kernel_function)


def get_cosine_kernel_matrix():
    return get_kernel_matrix_from_kernel_function(cosine_kernel_function)


def get_cosine_rbf_kernel_matrix():
    return get_kernel_matrix_from_kernel_function(cosine_rbf_kernel_function)


def get_cosine_distance_exponential_kernel_matrix():
    return get_kernel_matrix_from_kernel_function(
        cosine_distance_exponential_kernel_function
    )


def get_gaussian_rbf_euclidean_kernel_matrix(lengthscale):
    return get_kernel_matrix_from_kernel_function(
        gaussian_rbf_euclidean_kernel_function, lengthscale
    )


class FixedPrecomputedGPKernel(gpflow.kernels.Kernel):
    """
    A GPflow-compatible kernel that uses a precomputed, fixed kernel matrix.
    """

    def __init__(self, kernel_matrix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_matrix = kernel_matrix

    def K(self, X, X2=None):
        if not isinstance(X, np.ndarray):
            X = X.numpy()  # type: ignore

        X = X.flatten().astype(int)

        if X2 is None:
            indices = np.ix_(X, X)
        else:
            if not isinstance(X2, np.ndarray):
                X2 = X2.numpy()  # type: ignore

            X2 = X2.flatten().astype(int)

            indices = np.ix_(X, X2)
        return self.kernel_matrix[indices]

    def K_diag(self, X):  # type: ignore
        return np.ones(X.shape[0])  # type: ignore
