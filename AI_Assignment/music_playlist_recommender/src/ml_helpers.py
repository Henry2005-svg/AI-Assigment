import numpy as np


try:
    from sklearn.metrics import mean_squared_error as sklearn_mean_squared_error
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
except ModuleNotFoundError:
    sklearn_mean_squared_error = None
    sklearn_cosine_similarity = None
    sklearn_train_test_split = None


def cosine_similarity(matrix):
    """Use scikit-learn cosine similarity when installed, otherwise use a small NumPy version."""
    if sklearn_cosine_similarity is not None:
        return sklearn_cosine_similarity(matrix)

    dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = dense_matrix / norms
    return normalized @ normalized.T


def mean_squared_error(actual_values, predicted_values):
    """Use scikit-learn MSE when installed, otherwise calculate it with NumPy."""
    if sklearn_mean_squared_error is not None:
        return sklearn_mean_squared_error(actual_values, predicted_values)

    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    return np.mean((actual - predicted) ** 2)


def train_test_split(dataframe, test_size=0.2, random_state=42):
    """Use scikit-learn train_test_split when installed, otherwise split with pandas sampling."""
    if sklearn_train_test_split is not None:
        return sklearn_train_test_split(dataframe, test_size=test_size, random_state=random_state)

    test_df = dataframe.sample(frac=test_size, random_state=random_state)
    train_df = dataframe.drop(test_df.index)
    return train_df, test_df
