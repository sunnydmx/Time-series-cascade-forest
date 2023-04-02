import numpy as np
from numba import jit
from pyts.approximation import MultipleCoefficientBinning
from typing import Tuple


def filtering(sequence: np.array, window_size: int) -> np.array:
    '''
    This function mean filtering to reduce noise by using simple mean

    Params:
        sequence: 1d time series
        window_size: window_size for mean. It must be even

    Returns:
        np.ndarray: smoothed 1d time series
    '''
    assert (window_size - 1) % 2 == 0, 'window_size must be odds'
    assert (sequence.ndim) == 1, 'time series must be 1d array'

    half_window_size = (window_size - 1) // 2

    # i < w/2
    _left_values = sequence[:window_size - 1]
    _left_counts = np.ones(window_size - 1)
    _left_means = _left_values.cumsum() / _left_counts.cumsum()
    left_means = _left_means[half_window_size:]

    # w/2 < i < n - w/2
    slided_time_series = np.lib.stride_tricks.sliding_window_view(
        sequence, window_size
    )

    # n - w/2 < i
    _right_values = sequence[-window_size + 1:]
    _right_counts = np.ones(window_size - 1)
    _right_means = _right_values.cumsum() / _right_counts.cumsum()[::-1]
    right_means = _right_means[:-half_window_size]

    return np.concatenate([left_means, slided_time_series.mean(axis=1), right_means])


def slide_sequence(sequence: np.ndarray, num_neighbors: int) -> np.ndarray:
    '''
    This function means sliding sequence with num_neighbors for feature encoding.  

    Params:
        sequence: 1d time series
        num_neighbors: how many adjacent points are considered when slided

    Returns:
        np.ndarray: 2d time series
    '''
    assert (num_neighbors - 1) % 2 == 0

    half_num_neighbors = (num_neighbors - 1) // 2

    slided_sequence = np.lib.stride_tricks.sliding_window_view(sequence, num_neighbors)
    padded_slided_sequence = np.pad(
        slided_sequence,
        pad_width=((half_num_neighbors, half_num_neighbors), (0, 0)),
        mode='edge'
    )

    return padded_slided_sequence


def local_characteristics_of_sequence(sequence: np.ndarray) -> np.ndarray:
    '''
    This function mean the slope of time series to represent local characteristics of sequence.

    Params:
        sequence: time series

    Returns:
        np.ndarary: the slopes of two points
    '''
    if isinstance(sequence, np.ndarray):
        return np.diff(sequence)


def feature_coding(time_series: np.ndarray, num_alphabets: int) -> np.ndarray:
    '''
    This function mean feature_coding by using locally slope

    Params:
        time_series: 2d time series
        l: num of neighbors centering on each time point
        k: num of alphabet for feature coding

    Returns:
        np.ndarary: feature coding matrix of n * (l-1)
    '''
    assert 1 <= num_alphabets <= 26, 'num of alphabets is ranging from 1 to 26'

    multiple_coefficient_binning = MultipleCoefficientBinning(
        n_bins=num_alphabets,
        strategy='uniform'
    )
    feature_coding_matrix = multiple_coefficient_binning.fit_transform(time_series)

    return feature_coding_matrix


@jit(nopython=True)
def similarity(sequence1: np.ndarray, sequence2: np.ndarray):
    '''
    This function means the local shape similarity measures between feature coding matrix

    Params:
        sequence1: 1d time series
        sequence2: 1d time series

    Returns:
        float: similarity between two sequences
    '''
    _sum = 0.0
    n_row = len(sequence1)
    # print(n_row)
    for i in range(n_row):
        _sum += np.abs(sequence1[i] - sequence2[i])
    return _sum


@jit(nopython=True)
def _distance(
        sequence1: np.ndarray,
        sequence2: np.ndarray,
        feature_coding_matrix1: np.ndarray,
        feature_coding_matrix2: np.ndarray,
        alpha: float
):
    '''
    This function means the local shape similarity measures between feature coding matrix

    Params:
        sequence1: 1d time series
        sequence2: 1d time series

    Returns:
        distances: all distances calculated by dtw
        distances[n_row-1][n_col-1]: similarity between two sequences
    '''
    n_row = len(sequence1)
    n_col = len(sequence2)

    distances = np.ones((n_row, n_col))

    # first value of warping path
    # distance1 = sum(np.abs(feature_coding_matrix1[0] - feature_coding_matrix2[0]))
    distance1 = similarity(feature_coding_matrix1[0], feature_coding_matrix2[0])
    distance2 = (sequence1[0] - sequence2[0]) ** 2
    distances[0][0] = alpha * distance1 + (1 - alpha) * distance2

    # first column of warping path
    for i in range(1, n_row):
        # distance1 = sum(np.abs(feature_coding_matrix1[i] - feature_coding_matrix2[0]))
        distance1 = similarity(feature_coding_matrix1[i], feature_coding_matrix2[0])
        distance2 = (sequence1[i] - sequence2[0]) ** 2
        distances[i][0] = distances[i - 1][0] + (alpha * distance1 + (1 - alpha) * distance2)

    # top row of warping path
    for j in range(1, n_col):
        # distance1 = sum(np.abs(feature_coding_matrix1[0] - feature_coding_matrix2[j]))
        distance1 = similarity(feature_coding_matrix1[0], feature_coding_matrix2[j])
        distance2 = (sequence1[0] - sequence2[j]) ** 2
        distances[0][j] = distances[0][j - 1] + (alpha * distance1 + (1 - alpha) * distance2)

    # warping path
    for i in range(1, n_row):
        for j in range(1, n_col):
            # distance1 = sum(np.abs(feature_coding_matrix1[i] - feature_coding_matrix2[j]))
            distance1 = similarity(feature_coding_matrix1[i], feature_coding_matrix2[j])
            distance2 = (sequence1[i] - sequence2[j]) ** 2
            min_distance = min([distances[i - 1][j - 1], distances[i - 1][j], distances[i][j - 1]])

            distances[i][j] = min_distance + (alpha * distance1 + (1 - alpha) * distance2)

    return distances[n_row - 1][n_col - 1]


@jit(nopython=True)
def find_path(i: int, j: int, distances: np.array, epsilon=1e-10) -> Tuple[tuple, float]:
    '''
    This function mean find optimized path on the distances from LSDTW

    Params:
        i: the end of row index
        j: the end of column index
        distances: distances calculated by lsdtw

    Returns:
        warping_path: optimized path on the basis of distances
    '''
    start_point_of_i = i
    start_point_of_j = j
    warping_path = []

    while (i != 0) or (j != 0):
        if (i > 0 and j > 0):
            min_distance = min([distances[i - 1][j], distances[i - 1][j - 1], distances[i][j - 1]])

            if abs(distances[i - 1][j - 1] - min_distance) < epsilon:
                i -= 1
                j -= 1
            elif abs(distances[i - 1][j] - min_distance) < epsilon:
                i -= 1
            elif abs(distances[i][j - 1] - min_distance) < epsilon:
                j -= 1

        elif j > 0:
            j -= 1
        elif i > 0:
            i -= 1

        warping_path.append((i, j))

    warping_path = warping_path[::-1]
    warping_path.append((start_point_of_i, start_point_of_j))

    return warping_path
