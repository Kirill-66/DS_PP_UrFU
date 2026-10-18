import numpy as np
from grader_contracts.python_basics import (
    PositiveIntegerInput, TextInput, VectorPairInput,
)
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, MatrixInput, OneHotInput, RandomMatrixInput,
    TimeSeriesInput,
)
import python_basics as pb
import numpy_tasks as nt


assert pb.count_vowels(TextInput("Hello World")) == 3
assert pb.count_vowels(TextInput("")) == 0
assert pb.count_vowels(TextInput("AEIOU")) == 5
assert pb.count_vowels(TextInput("xyz")) == 0
assert pb.count_vowels(TextInput("AaEe")) == 4

assert pb.has_unique_characters(TextInput("abc")) is True
assert pb.has_unique_characters(TextInput("aba")) is False
assert pb.has_unique_characters(TextInput("")) is True
assert pb.has_unique_characters(TextInput("aA")) is True

assert pb.count_one_bits(PositiveIntegerInput(1)) == 1
assert pb.count_one_bits(PositiveIntegerInput(255)) == 8
assert pb.count_one_bits(PositiveIntegerInput(1024)) == 1
assert pb.count_one_bits(PositiveIntegerInput(3)) == 2

assert pb.multiplicative_persistence(PositiveIntegerInput(39)) == 3
assert pb.multiplicative_persistence(PositiveIntegerInput(4)) == 0
assert pb.multiplicative_persistence(PositiveIntegerInput(999)) == 4
assert pb.multiplicative_persistence(PositiveIntegerInput(10)) == 1
assert pb.multiplicative_persistence(PositiveIntegerInput(77)) == 4

assert pb.mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
assert pb.mse(VectorPairInput([0, 0], [3, 4])) == 12.5

assert pb.prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
assert pb.prime_factorization(PositiveIntegerInput(2)) == "(2)"
assert pb.prime_factorization(PositiveIntegerInput(100)) == "(2**2)(5**2)"
assert pb.prime_factorization(PositiveIntegerInput(97)) == "(97)"
assert pb.prime_factorization(PositiveIntegerInput(1)) == ""

assert pb.pyramid(PositiveIntegerInput(1)) == 1
assert pb.pyramid(PositiveIntegerInput(5)) == 2
assert pb.pyramid(PositiveIntegerInput(14)) == 3
assert pb.pyramid(PositiveIntegerInput(30)) == 4
assert pb.pyramid(PositiveIntegerInput(2)) == "It is impossible"
assert pb.pyramid(PositiveIntegerInput(8)) == "It is impossible"

assert pb.is_balanced_number(PositiveIntegerInput(1234006)) is True
assert pb.is_balanced_number(PositiveIntegerInput(123456)) is False
assert pb.is_balanced_number(PositiveIntegerInput(11)) is True

A = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]], dtype=float)
x = np.array([[[1], [1]], [[1], [1]]], dtype=float)
res = nt.sum_prod(type("D", (), {"matrices": A, "vectors": x})())
expected = A[0] @ x[0] + A[1] @ x[1]
assert res.shape == (2, 1)
assert np.allclose(res, expected)

A1 = np.array([[[1, 2], [3, 4]]], dtype=float)
x1 = np.array([[[1], [1]]], dtype=float)
res1 = nt.sum_prod(type("D", (), {"matrices": A1, "vectors": x1})())
assert res1.shape == (2, 1)
assert np.allclose(res1, A1[0] @ x1[0])

M = np.array([[0.1, 0.9], [0.5, 0.6]])
M_before = M.copy()
b = nt.binarize(BinarizeInput(matrix=M, threshold=0.5))
assert b.tolist() == [[0, 1], [0, 1]]
assert np.array_equal(M, M_before)

b2 = nt.binarize(BinarizeInput(matrix=np.array([[0.5, 0.51]]), threshold=0.5))
assert b2.tolist() == [[0, 1]]

ur = nt.unique_rows(MatrixInput(np.array([[1, 1, 2], [3, 3, 3]])))
assert [list(r) for r in ur] == [[1, 2], [3]]
uc = nt.unique_columns(MatrixInput(np.array([[1, 2], [1, 3]])))
assert [list(c) for c in uc] == [[1], [2, 3]]

stats = nt.matrix_statistics(RandomMatrixInput(rows=4, columns=5, seed=42))
assert stats.matrix.shape == (4, 5)
assert stats.row_means.shape == (4,)
assert stats.column_means.shape == (5,)
assert stats.row_variances.shape == (4,)
assert stats.column_variances.shape == (5,)
assert np.allclose(stats.row_means, stats.matrix.mean(axis=1))
assert np.allclose(stats.column_means, stats.matrix.mean(axis=0))
assert np.allclose(stats.row_variances, stats.matrix.var(axis=1))
assert np.allclose(stats.column_variances, stats.matrix.var(axis=0))

stats_again = nt.matrix_statistics(RandomMatrixInput(rows=4, columns=5, seed=42))
assert np.array_equal(stats.matrix, stats_again.matrix)

stats_custom = nt.matrix_statistics(
    RandomMatrixInput(rows=100, columns=100, mean=10, std=0, seed=7)
)
assert np.allclose(stats_custom.matrix, 10)

ch = nt.chess(ChessInput(rows=3, columns=4, first=0, second=1))
assert ch.tolist() == [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]

ch1 = nt.chess(ChessInput(rows=1, columns=1, first=7, second=9))
assert ch1.tolist() == [[7]]

img = nt.draw_rectangle(type("D", (), {
    "width": 2, "height": 2,
    "image_height": 5, "image_width": 5,
    "shape_color": (255, 0, 0), "background_color": (0, 0, 0),
})())
assert img.shape == (5, 5, 3)
assert img.dtype == np.uint8
assert tuple(img[2, 2]) == (255, 0, 0)
assert tuple(img[0, 0]) == (0, 0, 0)

ell = nt.draw_ellipse(type("D", (), {
    "semi_axis_x": 2, "semi_axis_y": 1,
    "image_height": 7, "image_width": 7,
    "shape_color": (0, 255, 0), "background_color": (0, 0, 0),
})())
assert ell.shape == (7, 7, 3)
assert tuple(ell[3, 3]) == (0, 255, 0)
assert tuple(ell[0, 0]) == (0, 0, 0)

ts_values = [1, 3, 2, 4, 1]
ts = nt.analyze_time_series(TimeSeriesInput(values=np.array(ts_values), window=2))
assert ts.local_maxima_indices.tolist() == [1, 3]
assert ts.local_minima_indices.tolist() == [2]
assert np.isclose(ts.mean, np.mean(ts_values))
assert np.isclose(ts.variance, np.var(ts_values))
assert np.isclose(ts.std, np.std(ts_values))
assert np.allclose(ts.moving_average, [2, 2.5, 3, 2.5])

ts_mono = nt.analyze_time_series(TimeSeriesInput(values=np.array([1, 2, 3, 4, 5]), window=2))
assert ts_mono.local_maxima_indices.size == 0
assert ts_mono.local_minima_indices.size == 0

ts_plateau = nt.analyze_time_series(TimeSeriesInput(values=np.array([1, 3, 3, 1]), window=2))
assert ts_plateau.local_maxima_indices.size == 0
assert ts_plateau.local_minima_indices.size == 0

x4 = np.array([1.0, 2.0, 3.0, 4.0])
ts_w1 = nt.analyze_time_series(TimeSeriesInput(values=x4, window=1))
assert np.allclose(ts_w1.moving_average, x4)
ts_wl = nt.analyze_time_series(TimeSeriesInput(values=x4, window=4))
assert ts_wl.moving_average.shape == (1,)
assert np.isclose(ts_wl.moving_average[0], 2.5)

oh = nt.one_hot(OneHotInput(labels=np.array([0, 2, 3, 0])))
assert oh.tolist() == [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]

oh2 = nt.one_hot(OneHotInput(labels=np.array([0, 1]), class_count=4))
assert oh2.shape == (2, 4)
assert oh2.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0]]

oh3 = nt.one_hot(OneHotInput(labels=np.array([2, 0]), class_count=4))
assert oh3.shape == (2, 4)
assert oh3.tolist() == [[0, 0, 1, 0], [1, 0, 0, 0]]

oh_empty = nt.one_hot(OneHotInput(labels=np.array([], dtype=int), class_count=4))
assert oh_empty.shape == (0, 4)

oh_empty_auto = nt.one_hot(OneHotInput(labels=np.array([], dtype=int)))
assert oh_empty_auto.shape == (0, 0)

print("all smoke tests passed")